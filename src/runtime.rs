//! Asynchronous, bounded effect boundary with keyed record/replay and
//! fsynced effect receipts. World reducers never perform model/network I/O.
use crate::core::AgentId;
use crate::effects::{EffectCallKey, EffectResult, ModelRequest, Sampling, Usage};
use crate::storage::{digest, stable_bytes};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::{self, OpenOptions}, future::Future, io::Write,
    path::{Path,PathBuf}, pin::Pin, sync::Arc, time::Duration};
use tokio::{sync::Semaphore, task::JoinSet, time::timeout};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag="type", content="value", rename_all="snake_case")]
pub enum EffectSpec {
    Model(ModelRequest),
    Tool { name: String, input: serde_json::Value },
}

impl EffectSpec {
    pub fn key(&self, trial: &str, step: u64, agent: AgentId, index: u32) -> Result<EffectCallKey,String> {
        Ok(EffectCallKey {
            trial_id: trial.into(), logical_step: step, agent_id: agent,
            call_index: index, canonical_request_hash: digest(self)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub key: EffectCallKey,
    pub spec: EffectSpec,
    pub result: EffectResult,
    /// Failure of world validation cannot reverse a provider charge.
    pub charged: bool,
    pub effective_provider: String,
    pub effective_model: Option<String>,
}

pub type BoxResult<'a> = Pin<Box<dyn Future<Output = Result<EffectResult,String>> + Send + 'a>>;
pub trait Backend: Send + Sync {
    fn resolve<'a>(&'a self, request: &'a EffectSpec) -> BoxResult<'a>;
}

/// Deterministic offline backend. Tools have no implicit executor.
#[derive(Default)]
pub struct ScriptedBackend;
impl Backend for ScriptedBackend {
    fn resolve<'a>(&'a self, spec: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move {
            match spec {
                EffectSpec::Model(model) => {
                    let text = model.messages.iter().filter(|m| matches!(m.role, crate::effects::Role::User))
                        .map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
                    Ok(EffectResult::Model {
                        content: format!("scripted:{text}"), effective_model: model.model.clone(),
                        usage: Some(Usage { input_tokens: None, output_tokens: None,
                            billed_cost: Some("0".into()), currency: Some("USD".into()) }),
                    })
                }
                EffectSpec::Tool {name, ..} => Ok(EffectResult::UnsupportedCapability {
                    capability: format!("no scripted implementation for tool {name}"),
                }),
            }
        })
    }
}

/// Replay from a validated receipt map by *full logical key* and request content.
/// Never assume completion-order cursors.
#[derive(Clone, Default)]
pub struct ReplayBackend {
    pub captured: BTreeMap<EffectCallKey, Receipt>,
}
impl ReplayBackend {
    pub fn from_receipts(receipts: impl IntoIterator<Item=Receipt>) -> Self {
        Self { captured: receipts.into_iter().map(|r| (r.key.clone(),r)).collect() }
    }
    pub fn resolve_key(&self, key: &EffectCallKey, spec: &EffectSpec) -> Result<Receipt,String> {
        let existing = self.captured.get(key).ok_or("missing keyed replay effect")?;
        if &existing.spec != spec || spec.key(&key.trial_id, key.logical_step, key.agent_id, key.call_index)? != *key {
            return Err("keyed replay request mismatch".into());
        }
        Ok(existing.clone())
    }
}

pub struct EffectJournal {
    pub path: PathBuf,
    pub receipts: BTreeMap<EffectCallKey, Receipt>,
    poisoned: bool,
}

impl EffectJournal {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self,String> {
        let path=dir.as_ref().join("effects.jsonl");
        if !path.exists() { OpenOptions::new().create_new(true).write(true).open(&path)
            .and_then(|f| f.sync_all()).map_err(|e| e.to_string())?; }
        let raw=fs::read(&path).map_err(|e| e.to_string())?;
        let length=raw.iter().rposition(|c| *c==b'\n').map(|i|i+1).unwrap_or(0);
        let mut receipts=BTreeMap::new();
        for line in raw[..length].split(|c|*c==b'\n').filter(|x|!x.is_empty()) {
            let receipt:Receipt=serde_json::from_slice(line).map_err(|e|format!("invalid effect ledger: {e}"))?;
            let key=&receipt.key;
            if receipt.spec.key(&key.trial_id,key.logical_step,key.agent_id,key.call_index)?!=*key
                || receipts.insert(key.clone(),receipt).is_some() {
                return Err("duplicate/mismatched effect ledger key".into());
            }
        }
        if length<raw.len() {
            let f=OpenOptions::new().write(true).open(&path).map_err(|e|e.to_string())?;
            f.set_len(length as u64).and_then(|_| f.sync_all()).map_err(|e|e.to_string())?;
        }
        Ok(Self{path,receipts,poisoned:false})
    }

    fn append(&mut self, receipt: Receipt)->Result<(),String>{
        if self.poisoned { return Err("effect ledger poisoned, reopen to recover".into()); }
        if self.receipts.contains_key(&receipt.key) { return Err("effect already recorded".into()); }
        let mut bytes=stable_bytes(&receipt)?; bytes.push(b'\n');
        if let Err(e)=(||->std::io::Result<()> {
            let mut f=OpenOptions::new().append(true).open(&self.path)?;
            f.write_all(&bytes)?; f.sync_all()
        })() { self.poisoned=true; return Err(format!("effect receipt fsync failure: {e}")); }
        self.receipts.insert(receipt.key.clone(),receipt);
        Ok(())
    }

    /// Concurrent model/tool requests; receipts are durable before returning to
    /// the deterministic reducer. Output order follows logical call keys.
    pub async fn resolve_all(
        &mut self, specs:Vec<(EffectCallKey,EffectSpec)>,
        backend:Arc<dyn Backend>, concurrency:usize, deadline:Duration,
        replay_only:bool,
    )->Result<Vec<Receipt>,String> {
        if concurrency==0 { return Err("concurrency limit must be positive".into()); }
        let mut seen=std::collections::BTreeSet::new();
        let sem=Arc::new(Semaphore::new(concurrency));
        let mut jobs=JoinSet::new();
        let mut finished=Vec::new();
        for (key,spec) in specs {
            if !seen.insert(key.clone()) { return Err("duplicate effect call key".into()); }
            if key!=spec.key(&key.trial_id,key.logical_step,key.agent_id,key.call_index)? {
                return Err("effect call hash mismatch".into());
            }
            if let Some(existing)=self.receipts.get(&key) {
                if existing.spec!=spec {return Err("recorded request content mismatch".into());}
                finished.push(existing.clone());
            } else {
                if replay_only {return Err(format!("missing recorded effect: {}", key.canonical_request_hash));}
                let sem=sem.clone();let backend=backend.clone();
                jobs.spawn(async move {
                    let permit=sem.acquire_owned().await.map_err(|e|e.to_string())?;
                    let result=timeout(deadline,backend.resolve(&spec)).await;
                    drop(permit);
                    let response=match result {
                        Ok(Ok(result))=>result,
                        Ok(Err(error))=>EffectResult::Failed{category:"backend".into(),message:error},
                        Err(_)=>EffectResult::Failed{category:"timeout".into(),message:"effect deadline exceeded".into()},
                    };
                    let (provider,model)=match &spec {
                        EffectSpec::Model(m)=>(m.provider.clone(),Some(m.model.clone())),
                        EffectSpec::Tool{name,..}=>(format!("tool:{name}"),None),
                    };
                    let charged=matches!(response,EffectResult::Model{..});
                    Ok::<_,String>(Receipt{key,spec,result:response,charged,
                        effective_provider:provider,effective_model:model})
                });
            }
        }
        while let Some(result)=jobs.join_next().await {
            let receipt=result.map_err(|e|format!("effect task failure: {e}"))??;
            self.append(receipt.clone())?;
            finished.push(receipt);
        }
        finished.sort_by(|a,b|a.key.cmp(&b.key));
        Ok(finished)
    }
}

/// Minimal OpenAI-compatible, opt-in JSON chat adapter, HTTP only via reqwest.
/// No network requests occur unless the "live-http" feature and backend are
/// selected explicitly. Provider seed remains best-effort, not D1.
#[cfg(feature="live-http")]
pub struct OpenAiCompatible {
    client:reqwest::Client,
    url:String,
    api_key:String,
}
#[cfg(feature="live-http")]
impl OpenAiCompatible {
    pub fn new(url:String,api_key:String)->Result<Self,String>{
        if !(url.starts_with("https://") || url.starts_with("http://127.0.0.1:")
          || url.starts_with("http://localhost:")) {
            return Err("live URL must use HTTPS or loopback HTTP".into());
        }
        Ok(Self{client:reqwest::Client::new(),url,api_key})
    }
}
#[cfg(feature="live-http")]
impl Backend for OpenAiCompatible {
    fn resolve<'a>(&'a self,spec:&'a EffectSpec)->BoxResult<'a> {
        Box::pin(async move {
            let model=match spec {
                EffectSpec::Model(m)=>m,
                _=>return Ok(EffectResult::UnsupportedCapability{capability:"HTTP tool".into()}),
            };
            if model.streaming || !model.tool_schemas.is_empty() || model.response_format.is_some() {
                return Ok(EffectResult::UnsupportedCapability{capability:"streaming/tool schemas/response format (minimal adapter)".into()});
            }
            let roles:Vec<_>=model.messages.iter().map(|m| {
                let role=match m.role {
                    crate::effects::Role::System=>"system",crate::effects::Role::User=>"user",
                    crate::effects::Role::Assistant=>"assistant",crate::effects::Role::Tool=>"tool",
                };
                serde_json::json!({"role":role,"content":m.content})
            }).collect();
            let mut body=serde_json::json!({"model":model.model,"messages":roles,"stream":false});
            if let Some(t)=model.sampling.temperature {body["temperature"]=serde_json::json!(t);}
            if let Some(p)=model.sampling.top_p {body["top_p"]=serde_json::json!(p);}
            if let Some(seed)=model.sampling.seed {body["seed"]=serde_json::json!(seed);}
            if let Some(max)=model.sampling.max_output_tokens {body["max_tokens"]=serde_json::json!(max);}
            let response=self.client.post(&self.url).bearer_auth(&self.api_key).json(&body).send()
                .await.map_err(|e|e.to_string())?;
            if !response.status().is_success() {
                return Err(format!("HTTP backend status {}",response.status()));
            }
            let payload:serde_json::Value=response.json().await.map_err(|e|e.to_string())?;
            let text=payload.pointer("/choices/0/message/content")
                .and_then(|v|v.as_str()).ok_or("missing chat completion content")?;
            let effective_model=payload.get("model").and_then(|v|v.as_str())
                .unwrap_or(model.model.as_str()).to_owned();
            let usage=payload.get("usage").map(|u|Usage {
                input_tokens:u.get("prompt_tokens").and_then(|v|v.as_u64()),
                output_tokens:u.get("completion_tokens").and_then(|v|v.as_u64()),
                billed_cost:None,currency:None
            });
            Ok(EffectResult::Model{content:text.into(),effective_model,usage})
        })
    }
}

/// Explicit scripted example request, used for independent-world tests.
pub fn sample_request(message:&str) -> EffectSpec {
    EffectSpec::Model(ModelRequest {
        provider:"scripted".into(),model:"fixed-v1".into(),resolved_model_version:Some("fixed-v1".into()),
        messages:vec![crate::effects::Message {
            role:crate::effects::Role::User, content:message.into(), tool_call_id:None
        }],
        sampling:Sampling{temperature:Some(0.0),top_p:None,seed:Some(1),max_output_tokens:Some(64)},
        tool_schemas:vec![],streaming:false,response_format:None,
    })
}
