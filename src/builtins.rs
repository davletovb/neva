//! Registered built-in deterministic counter society.
use crate::{core::{AgentId, World},experiment::EffectPolicy,runtime::{Receipt,EffectSpec,sample_request}};
use serde::{Serialize,Deserialize};

pub const KIND:&str="builtin.counter";
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Counter {
    pub total:u64,
}
impl World for Counter {
    type Action=u64;
    type Observation=u64;
    fn observe(&self,_:AgentId)->Result<u64,String>{Ok(self.total)}
    fn apply(&mut self,_:AgentId,add:u64)->Result<(),String>{
        self.total=self.total.checked_add(add).ok_or("counter overflow")?;
        Ok(())
    }
}
pub struct Increment {
    pub amount:u64,
    pub model:bool,
}
impl EffectPolicy<Counter> for Increment {
    fn requests(&self,actor:AgentId,obs:&u64)->Result<Vec<EffectSpec>,String>{
        if self.model {Ok(vec![sample_request(&format!("actor={} observed={obs}",actor.0))])}
        else {Ok(vec![])}
    }
    fn decide(&self,_:AgentId,_:&u64,receipts:&[Receipt])->Result<u64,String>{
        if self.model && receipts.len()!=1 {return Err("exactly one model result required".into());}
        Ok(self.amount)
    }
}
pub fn kind(amount:u64,model:bool)->String{
    format!("{KIND}:amount={amount}:model={model}")
}
pub fn parse_kind(input:&str)->Result<Increment,String>{
    let tail=input.strip_prefix(&format!("{KIND}:")).ok_or("unsupported built-in world")?;
    let mut parts=tail.split(':');
    let amount=parts.next().ok_or("missing amount")?.strip_prefix("amount=")
        .ok_or("invalid amount")?.parse::<u64>().map_err(|e|e.to_string())?;
    let model=parts.next().ok_or("missing model")?.strip_prefix("model=")
        .ok_or("invalid model")?.parse::<bool>().map_err(|e|e.to_string())?;
    if parts.next().is_some(){return Err("invalid world kind".into());}
    Ok(Increment{amount,model})
}
