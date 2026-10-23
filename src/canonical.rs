//! Direct writer for the stable JSON profile used by every Hiyal hash.
//!
//! Produces exactly the bytes of `serde_json::to_value` followed by
//! `core::stable_json_bytes` (recursively sorted object keys), without
//! building the intermediate `serde_json::Value` tree. Each object's entries
//! are written straight into one output buffer and reordered in place only
//! when serde emitted them out of order. Scalars and map keys are converted
//! by serde_json itself, so number, float and key handling cannot drift from
//! the reference profile, which `stable_bytes_via_value` keeps for tests.
//! So are serde_json's private special structs (`arbitrary_precision`
//! numbers, `raw_value` JSON), which features enabled anywhere in a build
//! can turn on through Cargo feature unification.
use serde::ser::{self, Serialize};
use std::{borrow::Cow, fmt::Display};

#[derive(Debug)]
pub(crate) struct Error(String);
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl ser::Error for Error {
    fn custom<T: Display>(msg: T) -> Self {
        Self(msg.to_string())
    }
}
impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

/// Stable JSON bytes of `value`, identical to the `serde_json::Value` route.
pub(crate) fn to_stable_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    value.serialize(Canonical(&mut out))?;
    Ok(out)
}

/// Reference implementation: build a `serde_json::Value`, then sort keys.
#[cfg(test)]
pub(crate) fn stable_bytes_via_value<T: Serialize + ?Sized>(
    value: &T,
) -> Result<Vec<u8>, serde_json::Error> {
    let value = serde_json::to_value(value)?;
    let mut out = Vec::new();
    crate::core::stable_json_bytes(&value, &mut out)?;
    Ok(out)
}

struct Canonical<'a>(&'a mut Vec<u8>);

/// Write a scalar exactly as the `Value` route would: convert it with
/// serde_json's own `Value` serializer, then write that value.
fn scalar<T: Serialize + ?Sized>(out: &mut Vec<u8>, value: &T) -> Result<(), Error> {
    let value = serde_json::to_value(value)?;
    serde_json::to_writer(out, &value)?;
    Ok(())
}

fn string(out: &mut Vec<u8>, value: &str) -> Result<(), Error> {
    serde_json::to_writer(out, value)?;
    Ok(())
}

/// `{"variant":` before a variant's content; the caller closes the brace.
fn open_variant(out: &mut Vec<u8>, variant: &str) -> Result<(), Error> {
    out.push(b'{');
    string(out, variant)?;
    out.push(b':');
    Ok(())
}

/// Map keys follow serde_json's map-key rules (integer and bool keys become
/// strings, unsupported key types are rejected) by serializing a one-entry
/// map to a `Value` and reading its only key back.
fn map_key<T: Serialize + ?Sized>(key: &T) -> Result<String, Error> {
    struct Probe<'a, T: ?Sized>(&'a T);
    impl<T: Serialize + ?Sized> Serialize for Probe<'_, T> {
        fn serialize<S: ser::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            use ser::SerializeMap;
            let mut map = s.serialize_map(Some(1))?;
            map.serialize_entry(self.0, &())?;
            map.end()
        }
    }
    match serde_json::to_value(Probe(key))? {
        serde_json::Value::Object(map) => map
            .into_iter()
            .next()
            .map(|(key, _)| key)
            .ok_or_else(|| Error("map key probe produced no key".into())),
        _ => Err(Error("map key probe produced no object".into())),
    }
}

struct Entry {
    key: Cow<'static, str>,
    /// Byte range of `"key":value` in the output, excluding any comma.
    start: usize,
    end: usize,
}

struct Object<'a> {
    out: &'a mut Vec<u8>,
    start: usize,
    entries: Vec<Entry>,
    pending_key: Option<String>,
    /// Extra `}` for struct variants, which wrap the object in another one.
    close_variant: bool,
}

impl<'a> Object<'a> {
    fn open(out: &'a mut Vec<u8>, close_variant: bool) -> Self {
        out.push(b'{');
        let start = out.len();
        Self {
            out,
            start,
            entries: Vec::new(),
            pending_key: None,
            close_variant,
        }
    }

    fn entry<T: Serialize + ?Sized>(
        &mut self,
        key: Cow<'static, str>,
        value: &T,
    ) -> Result<(), Error> {
        if !self.entries.is_empty() {
            self.out.push(b',');
        }
        let start = self.out.len();
        string(self.out, &key)?;
        self.out.push(b':');
        value.serialize(Canonical(self.out))?;
        let end = self.out.len();
        self.entries.push(Entry { key, start, end });
        Ok(())
    }

    /// Close the object. Keys are sorted byte-wise, matching `String`'s
    /// order; for a repeated key the last value wins, as in a `Value` map.
    fn finish(mut self) -> Result<(), Error> {
        let in_order = self
            .entries
            .windows(2)
            .all(|pair| pair[0].key.as_bytes() < pair[1].key.as_bytes());
        if !in_order {
            let written = self.out.split_off(self.start);
            let base = self.start;
            // Stable sort keeps insertion order among equal keys, so the
            // last of each run is the value a `Value` map would keep.
            self.entries
                .sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
            let mut first = true;
            for (index, entry) in self.entries.iter().enumerate() {
                if self
                    .entries
                    .get(index + 1)
                    .is_some_and(|next| next.key == entry.key)
                {
                    continue;
                }
                if !first {
                    self.out.push(b',');
                }
                first = false;
                self.out
                    .extend_from_slice(&written[entry.start - base..entry.end - base]);
            }
        }
        self.out.push(b'}');
        if self.close_variant {
            self.out.push(b'}');
        }
        Ok(())
    }
}

impl ser::SerializeMap for Object<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Error> {
        self.pending_key = Some(map_key(key)?);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        let key = self
            .pending_key
            .take()
            .ok_or_else(|| Error("map value without key".into()))?;
        self.entry(Cow::Owned(key), value)
    }
    fn end(self) -> Result<(), Error> {
        self.finish()
    }
}

/// serde_json encodes `arbitrary_precision` numbers and `raw_value` JSON as
/// structs with these reserved names; its `Value` serializer turns them back
/// into a number or parsed JSON rather than an object.
const SERDE_JSON_PRIVATE: &str = "$serde_json::private::";

enum StructSer<'a> {
    Object(Object<'a>),
    /// A serde_json special struct, converted by serde_json's own `Value`
    /// serializer and written with the reference key sort.
    Special {
        out: &'a mut Vec<u8>,
        name: &'static str,
        written: bool,
    },
}

impl ser::SerializeStruct for StructSer<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        match self {
            Self::Object(object) => object.entry(Cow::Borrowed(key), value),
            Self::Special { out, name, written } => {
                struct Special<'a, T: ?Sized> {
                    name: &'static str,
                    key: &'static str,
                    value: &'a T,
                }
                impl<T: Serialize + ?Sized> Serialize for Special<'_, T> {
                    fn serialize<S: ser::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                        use ser::SerializeStruct;
                        let mut special = s.serialize_struct(self.name, 1)?;
                        special.serialize_field(self.key, self.value)?;
                        special.end()
                    }
                }
                if *written {
                    return Err(Error("serde_json special struct has several fields".into()));
                }
                let value = serde_json::to_value(Special { name, key, value })?;
                crate::core::stable_json_bytes(&value, out)?;
                *written = true;
                Ok(())
            }
        }
    }
    fn end(self) -> Result<(), Error> {
        match self {
            Self::Object(object) => object.finish(),
            Self::Special { written: true, .. } => Ok(()),
            Self::Special { .. } => Err(Error("empty serde_json special struct".into())),
        }
    }
}

impl ser::SerializeStructVariant for Object<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.entry(Cow::Borrowed(key), value)
    }
    fn end(self) -> Result<(), Error> {
        self.finish()
    }
}

struct Seq<'a> {
    out: &'a mut Vec<u8>,
    first: bool,
    /// Extra `}` for tuple variants, which wrap the array in an object.
    close_variant: bool,
}

impl<'a> Seq<'a> {
    fn open(out: &'a mut Vec<u8>, close_variant: bool) -> Self {
        out.push(b'[');
        Self {
            out,
            first: true,
            close_variant,
        }
    }
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        if !self.first {
            self.out.push(b',');
        }
        self.first = false;
        value.serialize(Canonical(self.out))
    }
    fn finish(self) -> Result<(), Error> {
        self.out.push(b']');
        if self.close_variant {
            self.out.push(b'}');
        }
        Ok(())
    }
}

macro_rules! seq_impl {
    ($($trait:ident :: $method:ident),*) => {
        $(impl ser::$trait for Seq<'_> {
            type Ok = ();
            type Error = Error;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
                self.element(value)
            }
            fn end(self) -> Result<(), Error> {
                self.finish()
            }
        })*
    };
}
seq_impl!(
    SerializeSeq::serialize_element,
    SerializeTuple::serialize_element,
    SerializeTupleStruct::serialize_field,
    SerializeTupleVariant::serialize_field
);

macro_rules! scalar_methods {
    ($($method:ident: $ty:ty),* $(,)?) => {
        $(fn $method(self, value: $ty) -> Result<(), Error> {
            scalar(self.0, &value)
        })*
    };
}

impl<'a> ser::Serializer for Canonical<'a> {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Seq<'a>;
    type SerializeTuple = Seq<'a>;
    type SerializeTupleStruct = Seq<'a>;
    type SerializeTupleVariant = Seq<'a>;
    type SerializeMap = Object<'a>;
    type SerializeStruct = StructSer<'a>;
    type SerializeStructVariant = Object<'a>;

    scalar_methods! {
        serialize_bool: bool,
        serialize_i8: i8, serialize_i16: i16, serialize_i32: i32, serialize_i64: i64,
        serialize_i128: i128,
        serialize_u8: u8, serialize_u16: u16, serialize_u32: u32, serialize_u64: u64,
        serialize_u128: u128,
        serialize_f32: f32, serialize_f64: f64,
        serialize_char: char,
    }
    fn serialize_str(self, value: &str) -> Result<(), Error> {
        string(self.0, value)
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<(), Error> {
        // The `Value` serializer turns bytes into an array of numbers.
        let mut seq = Seq::open(self.0, false);
        for byte in value {
            seq.element(byte)?;
        }
        seq.finish()
    }
    fn serialize_none(self) -> Result<(), Error> {
        self.0.extend_from_slice(b"null");
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Error> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), Error> {
        self.serialize_none()
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Error> {
        self.serialize_none()
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), Error> {
        string(self.0, variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        open_variant(self.0, variant)?;
        value.serialize(Canonical(&mut *self.0))?;
        self.0.push(b'}');
        Ok(())
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Seq<'a>, Error> {
        Ok(Seq::open(self.0, false))
    }
    fn serialize_tuple(self, _: usize) -> Result<Seq<'a>, Error> {
        Ok(Seq::open(self.0, false))
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Seq<'a>, Error> {
        Ok(Seq::open(self.0, false))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Seq<'a>, Error> {
        open_variant(self.0, variant)?;
        Ok(Seq::open(self.0, true))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Object<'a>, Error> {
        Ok(Object::open(self.0, false))
    }
    fn serialize_struct(self, name: &'static str, _: usize) -> Result<StructSer<'a>, Error> {
        Ok(if name.starts_with(SERDE_JSON_PRIVATE) {
            StructSer::Special {
                out: self.0,
                name,
                written: false,
            }
        } else {
            StructSer::Object(Object::open(self.0, false))
        })
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Object<'a>, Error> {
        open_variant(self.0, variant)?;
        Ok(Object::open(self.0, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{RngCore, SeedableRng};
    use serde::Serialize;
    use serde_json::{Map, Value, json};
    use std::collections::{BTreeMap, HashMap};

    /// The new writer must agree with the `Value` route: same bytes, or both fail.
    fn same<T: Serialize + ?Sized>(value: &T) {
        let reference = stable_bytes_via_value(value);
        let direct = to_stable_bytes(value);
        match (reference, direct) {
            (Ok(a), Ok(b)) => assert_eq!(
                String::from_utf8_lossy(&a),
                String::from_utf8_lossy(&b),
                "byte mismatch"
            ),
            (Err(_), Err(_)) => {}
            (a, b) => panic!("outcome mismatch: reference {a:?}, direct {b:?}"),
        }
    }

    const KEY_CHARS: &[&str] = &[
        "a", "b", "B", "_", "0", "9", "é", "z", "\"", "\\", "\n", "\u{1}", "\u{7f}", "ß", "€",
        "😀", "\u{ffff}", " ", "/", "~",
    ];

    fn text(rng: &mut ChaCha20Rng, max: u32) -> String {
        (0..rng.next_u32() % max)
            .map(|_| KEY_CHARS[(rng.next_u32() as usize) % KEY_CHARS.len()])
            .collect()
    }

    fn float(rng: &mut ChaCha20Rng) -> f64 {
        match rng.next_u32() % 6 {
            0 => -0.0,
            1 => {
                f64::from_bits(rng.next_u64() & !(0x7ff << 52) | (rng.next_u64() % 2046 + 1) << 52)
            }
            2 => f64::MIN_POSITIVE * f64::from(rng.next_u32()),
            3 => f64::MAX / f64::from(rng.next_u32().max(1)),
            4 => (rng.next_u32() as f64) / 7.0,
            _ => 1e300 * (rng.next_u32() as f64),
        }
    }

    fn value(rng: &mut ChaCha20Rng, depth: u32) -> Value {
        let kinds = if depth == 0 { 6 } else { 8 };
        match rng.next_u32() % kinds {
            0 => Value::Null,
            1 => Value::Bool(rng.next_u32() & 1 == 0),
            2 => json!(rng.next_u64() as i64),
            3 => json!(rng.next_u64()),
            4 => serde_json::Number::from_f64(float(rng)).map_or(Value::Null, Value::Number),
            5 => Value::String(text(rng, 12)),
            6 => Value::Array(
                (0..rng.next_u32() % 5)
                    .map(|_| value(rng, depth - 1))
                    .collect(),
            ),
            _ => {
                let mut map = Map::new();
                for _ in 0..rng.next_u32() % 7 {
                    map.insert(text(rng, 4), value(rng, depth - 1));
                }
                Value::Object(map)
            }
        }
    }

    #[test]
    fn random_json_values_match_reference_bytes() {
        let mut rng = ChaCha20Rng::seed_from_u64(117);
        for _ in 0..20_000 {
            same(&value(&mut rng, 4));
        }
    }

    #[test]
    fn random_maps_with_typed_keys_match_reference_bytes() {
        let mut rng = ChaCha20Rng::seed_from_u64(118);
        for _ in 0..2_000 {
            let ints: HashMap<i64, f64> = (0..rng.next_u32() % 9)
                .map(|_| (rng.next_u64() as i64 % 1000, float(&mut rng)))
                .collect();
            same(&ints);
            let unsigned: BTreeMap<u128, Vec<u8>> = (0..rng.next_u32() % 5)
                .map(|_| (u128::from(rng.next_u64()), vec![rng.next_u32() as u8; 3]))
                .collect();
            same(&unsigned);
            let strings: HashMap<String, Option<i32>> = (0..rng.next_u32() % 9)
                .map(|_| (text(&mut rng, 5), Some(rng.next_u32() as i32)))
                .collect();
            same(&strings);
        }
    }

    #[derive(Serialize)]
    struct Unit;
    #[derive(Serialize)]
    struct Newtype(u8);
    #[derive(Serialize)]
    struct Tuple(i8, &'static str, f32);
    #[derive(Serialize)]
    enum Shapes {
        Unit,
        Newtype(Vec<u16>),
        Tuple(u8, char),
        Struct { zeta: bool, alpha: Option<f64> },
    }
    #[derive(Serialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Tagged {
        Plain { zebra: u8, apple: String },
        Wrapped(BTreeMap<String, u8>),
        Empty,
    }
    #[derive(Serialize)]
    #[serde(tag = "t", content = "c")]
    enum Adjacent {
        One(u8),
        Two { b: u8, a: u8 },
    }
    #[derive(Serialize)]
    struct Inner {
        shared: u8,
        later: &'static str,
    }
    #[derive(Serialize)]
    struct Everything {
        z_last: u64,
        a_first: i128,
        #[serde(rename = "Middle")]
        middle: u128,
        float32: f32,
        float64: f64,
        nan: f64,
        inf: f32,
        neg_zero: f64,
        character: char,
        unit: (),
        unit_struct: Unit,
        newtype: Newtype,
        tuple: Tuple,
        pair: (u8, i64),
        shapes: Vec<Shapes>,
        tagged: Vec<Tagged>,
        adjacent: Vec<Adjacent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        skipped: Option<u8>,
        absent: Option<u8>,
        bool_keys: BTreeMap<bool, u8>,
        char_keys: BTreeMap<char, u8>,
        #[serde(flatten)]
        flattened: Inner,
        #[serde(flatten)]
        extra: BTreeMap<String, u8>,
    }

    fn everything(a_first: i128, middle: u128) -> Everything {
        Everything {
            z_last: u64::MAX,
            a_first,
            middle,
            float32: 0.1,
            float64: 0.1,
            nan: f64::NAN,
            inf: f32::INFINITY,
            neg_zero: -0.0,
            character: '"',
            unit: (),
            unit_struct: Unit,
            newtype: Newtype(7),
            tuple: Tuple(-3, "t\u{0}", 1.5e-40),
            pair: (1, -1),
            shapes: vec![
                Shapes::Unit,
                Shapes::Newtype(vec![1, 2]),
                Shapes::Tuple(9, 'é'),
                Shapes::Struct {
                    zeta: true,
                    alpha: Some(f64::MIN_POSITIVE),
                },
                Shapes::Struct {
                    zeta: false,
                    alpha: None,
                },
            ],
            tagged: vec![
                Tagged::Plain {
                    zebra: 1,
                    apple: "x".into(),
                },
                Tagged::Wrapped(BTreeMap::from([("kind2".into(), 2), ("a".into(), 1)])),
                Tagged::Empty,
            ],
            adjacent: vec![Adjacent::One(1), Adjacent::Two { b: 2, a: 1 }],
            skipped: None,
            absent: None,
            bool_keys: BTreeMap::from([(true, 1), (false, 0)]),
            char_keys: BTreeMap::from([('b', 2), ('a', 1)]),
            flattened: Inner {
                shared: 1,
                later: "flat",
            },
            // Duplicates a flattened key: the last write must win.
            extra: BTreeMap::from([("shared".into(), 9), ("aa".into(), 0)]),
        }
    }

    #[test]
    fn every_serde_shape_matches_reference_bytes() {
        same(&everything(-5, 5));
        same(&everything(i128::from(i64::MIN), u128::from(u64::MAX)));
        // Outside serde_json's integer range: both routes reject it, or under
        // `arbitrary_precision` both accept it with the same bytes.
        same(&everything(i128::MAX, 0));
        same(&everything(0, u128::MAX));
    }

    #[test]
    fn unsupported_map_keys_and_bytes_match_reference() {
        struct Bytes(&'static [u8]);
        impl Serialize for Bytes {
            fn serialize<S: ser::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_bytes(self.0)
            }
        }
        same(&Bytes(b"\x00\xffabc"));
        let vec_keys: BTreeMap<Vec<u8>, u8> = BTreeMap::from([(vec![1], 1)]);
        same(&vec_keys);
        assert!(to_stable_bytes(&vec_keys).is_err());
        let float_keys: Vec<(f64, u8)> = vec![(1.5, 1)];
        same(&float_keys);
        let unit_keys: BTreeMap<(), u8> = BTreeMap::from([((), 1)]);
        same(&unit_keys);
        let tuple_keys: BTreeMap<(u8, u8), u8> = BTreeMap::from([((1, 2), 1)]);
        same(&tuple_keys);
        let newtype_keys: BTreeMap<Newtype2, u8> = BTreeMap::from([(Newtype2(3), 1)]);
        same(&newtype_keys);
    }
    #[derive(Serialize, PartialEq, Eq, PartialOrd, Ord)]
    struct Newtype2(u32);

    /// serde_json's special structs, written by hand so the test runs in every
    /// feature set. With `arbitrary_precision`/`raw_value` off both routes see
    /// ordinary structs; with them on both must convert them identically.
    struct Special(&'static str, &'static str);
    impl Serialize for Special {
        fn serialize<S: ser::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            use ser::SerializeStruct;
            let mut special = s.serialize_struct(self.0, 1)?;
            special.serialize_field(self.0, self.1)?;
            special.end()
        }
    }

    #[test]
    fn serde_json_special_structs_match_reference_bytes() {
        const NUMBER: &str = "$serde_json::private::Number";
        const RAW: &str = "$serde_json::private::RawValue";
        for (name, text) in [
            (NUMBER, "42"),
            (NUMBER, "-0.0"),
            (NUMBER, "123456789012345678901234567890"),
            (NUMBER, "1.000000000000000000001e-400"),
            (RAW, r#"{"b":[1,{"z":0,"a":1}],"a":"x"}"#),
            (RAW, "[3,2,1]"),
            (RAW, "12345678901234567890123"),
        ] {
            same(&Special(name, text));
            same(&vec![Special(name, text)]);
            same(&BTreeMap::from([("outer", Special(name, text))]));
        }
        same(&json!({"checkpoint": 200, "arm": "both", "big": 18446744073709551615u64}));
    }

    #[test]
    fn engine_records_match_reference_bytes() {
        let mut streams = crate::rng::Streams::new("trial".into(), 7);
        for name in ["scheduler", "agent/10", "agent/2", "world"] {
            streams.draw_u64(name).unwrap();
        }
        same(&streams);
        same(&streams.paired_branch("branch".into()));
        same(&crate::storage::BuildProvenance::current());
    }
}
