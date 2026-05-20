//! HDC (Hyperdimensional Computing) SQL function bridge.
//!
//! Provides SQL functions for encoding domain data into binary hypervectors,
//! computing similarity, and performing HDC algebra (bind, bundle).
//!
//! # SQL Functions
//!
//! - `HDC_ENCODE(domain, method, json)` — Encode domain data into a hypervector
//! - `HDC_ENCODE_TEXT(text [, dim])` — Fuzzy text encoding (character-level)
//! - `HDC_ENCODE_HASH(data [, dim])` — Hash-based encoding (whole-input)
//! - `HDC_SIMILARITY(hv1, hv2)` — Normalized similarity (0.0–1.0)
//! - `HDC_BIPOLAR_SIMILARITY(hv1, hv2)` — Bipolar similarity (-1.0–1.0)
//! - `HDC_DISTANCE(hv1, hv2)` — Hamming distance (integer)
//! - `HDC_BIND(hv1, hv2)` — XOR binding (association)
//! - `HDC_BUNDLE(hv1, hv2, ...)` — Majority-vote bundling (superposition)
//! - `HDC_DIMS(hv)` — Dimension count

use base64::Engine;
use joule_db_hdc::turbo_holographic::{BinaryHV, BundleAccumulator};
use std::cell::RefCell;
use std::collections::HashMap;

/// Default dimension for HDC vectors (matches domain encoder default).
pub const DEFAULT_DIMENSION: usize = 10000;

// ============================================================================
// Serialization: BinaryHV <-> TEXT
// ============================================================================

/// Serialize a BinaryHV to a text string: `hdc:<dimension>:<base64>`
///
/// The base64 payload encodes the packed u64 words as little-endian bytes.
pub fn serialize_hv(hv: &BinaryHV) -> String {
    let words = hv.as_words();
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let encoded = base64::engine::general_purpose::STANDARD_NO_PAD.encode(&bytes);
    format!("hdc:{}:{}", hv.dimension(), encoded)
}

/// Deserialize a text string back to a BinaryHV.
///
/// Expected format: `hdc:<dimension>:<base64>`
pub fn deserialize_hv(text: &str) -> Option<BinaryHV> {
    let parts: Vec<&str> = text.splitn(3, ':').collect();
    if parts.len() != 3 || parts[0] != "hdc" {
        return None;
    }
    let dimension: usize = parts[1].parse().ok()?;
    let bytes = base64::engine::general_purpose::STANDARD_NO_PAD
        .decode(parts[2])
        .ok()?;
    if bytes.len() % 8 != 0 {
        return None;
    }
    let words: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let expected_words = (dimension + 63) / 64;
    if words.len() != expected_words {
        return None;
    }
    Some(BinaryHV::from_words(words, dimension))
}

// ============================================================================
// Domain Encoder Trait + Thread-Local Pool
// ============================================================================

/// Trait for domain-specific HDC encoders callable from SQL.
trait DomainEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String>;
}

/// Result of a domain encode call — usually a BinaryHV, sometimes a string (e.g., trend direction).
enum EncoderResult {
    Vector(BinaryHV),
    Text(String),
}

thread_local! {
    static ENCODER_POOL: RefCell<HashMap<String, Box<dyn DomainEncoder>>> =
        RefCell::new(HashMap::new());
}

/// Encode domain data via the thread-local encoder pool.
///
/// Returns either a serialized BinaryHV string or a plain text result.
pub fn hdc_encode(domain: &str, method: &str, json_str: &str) -> Result<String, String> {
    let json: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid JSON: {}", e))?;

    let domain_lower = domain.to_lowercase();

    ENCODER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let encoder = pool
            .entry(domain_lower.clone())
            .or_insert_with(|| create_encoder(&domain_lower));
        match encoder.encode(method, &json)? {
            EncoderResult::Vector(hv) => Ok(serialize_hv(&hv)),
            EncoderResult::Text(s) => Ok(s),
        }
    })
}

/// Factory: create a domain encoder by name.
fn create_encoder(domain: &str) -> Box<dyn DomainEncoder> {
    match domain {
        "market" => Box::new(MarketEncoder::new()),
        "health" => Box::new(HealthEncoder::new()),
        "cyber" => Box::new(CyberEncoder::new()),
        "iot" => Box::new(IotEncoder::new()),
        "temporal" => Box::new(TemporalEncoder::new()),
        "media" => Box::new(MediaEncoder::new()),
        "genomics" => Box::new(GenomicsEncoder::new()),
        "legal" => Box::new(LegalEncoder::new()),
        "energy" => Box::new(EnergyEncoder::new()),
        "adtech" => Box::new(AdTechEncoder::new()),
        "agri" => Box::new(AgriEncoder::new()),
        "graph" => Box::new(GraphEncoder::new()),
        "edu" => Box::new(EduEncoder::new()),
        "supply" => Box::new(SupplyEncoder::new()),
        "retail" => Box::new(RetailEncoder::new()),
        "auto" => Box::new(AutoEncoder::new()),
        "gaming" => Box::new(GamingEncoder::new()),
        "insurance" => Box::new(InsuranceEncoder::new()),
        "spatial" => Box::new(SpatialEncoder::new()),
        "multimodal" => Box::new(MultimodalEncoder::new()),
        "telecom" => Box::new(TelecomEncoder::new()),
        _ => Box::new(UnknownEncoder(domain.to_string())),
    }
}

/// Helper: deserialize JSON to a struct, mapping serde errors to strings.
fn deser<T: serde::de::DeserializeOwned>(json: &serde_json::Value) -> Result<T, String> {
    serde_json::from_value(json.clone()).map_err(|e| format!("JSON parse error: {}", e))
}

/// Helper: extract f64 from JSON value.
fn as_f64(json: &serde_json::Value) -> Result<f64, String> {
    json.as_f64()
        .ok_or_else(|| "Expected a numeric value".to_string())
}

// ============================================================================
// Unknown domain (error on any method)
// ============================================================================

struct UnknownEncoder(String);
impl DomainEncoder for UnknownEncoder {
    fn encode(
        &mut self,
        _method: &str,
        _json: &serde_json::Value,
    ) -> Result<EncoderResult, String> {
        Err(format!("Unknown HDC domain: '{}'", self.0))
    }
}

// ============================================================================
// Market Domain
// ============================================================================

struct MarketEncoder {
    link: joule_db_domains::market::MarketLink,
}

impl MarketEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::market::MarketLink::new(),
        }
    }
}

impl DomainEncoder for MarketEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "trade" => {
                let t: joule_db_domains::market::Trade = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_trade(&t)))
            }
            "option_trade" => {
                let t: joule_db_domains::market::OptionTrade = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_option_trade(&t)))
            }
            "future_trade" => {
                let t: joule_db_domains::market::FutureTrade = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_future_trade(&t)))
            }
            "delta_probe" => Ok(EncoderResult::Vector(
                self.link.encode_delta_probe(as_f64(json)?),
            )),
            "gamma_probe" => Ok(EncoderResult::Vector(
                self.link.encode_gamma_probe(as_f64(json)?),
            )),
            "price_term" => Ok(EncoderResult::Vector(
                self.link.encode_price_term(as_f64(json)?),
            )),
            _ => Err(format!("Unknown market method: '{}'", method)),
        }
    }
}

// ============================================================================
// Health Domain
// ============================================================================

struct HealthEncoder {
    link: joule_db_domains::health::HealthLink,
}

impl HealthEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::health::HealthLink::new(),
        }
    }
}

impl DomainEncoder for HealthEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "patient" => {
                let p: joule_db_domains::health::Patient = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_patient(&p)))
            }
            "vitals" => {
                let v: joule_db_domains::health::Vitals = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_vitals(&v)))
            }
            "diagnosis" => {
                let d: joule_db_domains::health::Diagnosis = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_diagnosis(&d)))
            }
            "medication" => {
                let m: joule_db_domains::health::Medication = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_medication(&m)))
            }
            "lab_result" => {
                let l: joule_db_domains::health::LabResult = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_lab_result(&l)))
            }
            "symptom" => {
                let s: joule_db_domains::health::Symptom = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_symptom(&s)))
            }
            "procedure" => {
                let p: joule_db_domains::health::Procedure = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_procedure(&p)))
            }
            "encounter" => {
                let e: joule_db_domains::health::MedicalEncounter = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_encounter(&e)))
            }
            _ => Err(format!("Unknown health method: '{}'", method)),
        }
    }
}

// ============================================================================
// Cyber Domain
// ============================================================================

struct CyberEncoder {
    link: joule_db_domains::cyber::CyberLink,
}

impl CyberEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::cyber::CyberLink::new(),
        }
    }
}

impl DomainEncoder for CyberEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "ioc" => {
                let i: joule_db_domains::cyber::Ioc = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_ioc(&i)))
            }
            "malware" => {
                let m: joule_db_domains::cyber::MalwareSample = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_malware(&m)))
            }
            "flow" => {
                let f: joule_db_domains::cyber::NetworkFlow = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_flow(&f)))
            }
            "event" => {
                let e: joule_db_domains::cyber::SecurityEvent = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_event(&e)))
            }
            _ => Err(format!("Unknown cyber method: '{}'", method)),
        }
    }
}

// ============================================================================
// IoT Domain
// ============================================================================

struct IotEncoder {
    link: joule_db_domains::iot::IotLink,
}

impl IotEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::iot::IotLink::new(),
        }
    }
}

impl DomainEncoder for IotEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "reading" => {
                let r: joule_db_domains::iot::SensorReading = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_reading(&r)))
            }
            "sensor_fusion" => {
                let readings: Vec<joule_db_domains::iot::SensorReading> = deser(json)?;
                Ok(EncoderResult::Vector(
                    self.link.encode_sensor_fusion(&readings),
                ))
            }
            "asset" => {
                let a: joule_db_domains::iot::Asset = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_asset(&a)))
            }
            "alert" => {
                let a: joule_db_domains::iot::Alert = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_alert(&a)))
            }
            "maintenance" => {
                let m: joule_db_domains::iot::MaintenanceEvent = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_maintenance(&m)))
            }
            _ => Err(format!("Unknown iot method: '{}'", method)),
        }
    }
}

// ============================================================================
// Temporal Domain
// ============================================================================

struct TemporalEncoder {
    link: joule_db_domains::temporal::TemporalLink,
}

impl TemporalEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::temporal::TemporalLink::new(),
        }
    }
}

impl DomainEncoder for TemporalEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "window" => {
                let points: Vec<joule_db_domains::temporal::TimePoint> = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_window(&points)))
            }
            "trend" => {
                let points: Vec<joule_db_domains::temporal::TimePoint> = deser(json)?;
                let direction = self.link.encode_trend(&points);
                Ok(EncoderResult::Text(format!("{:?}", direction)))
            }
            _ => Err(format!("Unknown temporal method: '{}'", method)),
        }
    }
}

// ============================================================================
// Media Domain
// ============================================================================

struct MediaEncoder {
    link: joule_db_domains::media::MediaLink,
}

impl MediaEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::media::MediaLink::new(),
        }
    }
}

impl DomainEncoder for MediaEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "content" => {
                let c: joule_db_domains::media::Content = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_content(&c)))
            }
            "creator" => {
                let c: joule_db_domains::media::Creator = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_creator(&c)))
            }
            "viewer" => {
                let v: joule_db_domains::media::Viewer = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_viewer(&v)))
            }
            "engagement" => {
                let e: joule_db_domains::media::Engagement = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_engagement(&e)))
            }
            _ => Err(format!("Unknown media method: '{}'", method)),
        }
    }
}

// ============================================================================
// Genomics Domain (custom JSON for Base/DnaSequence/ProteinSequence)
// ============================================================================

struct GenomicsEncoder {
    link: joule_db_domains::genomics::GenomicsLink,
}

impl GenomicsEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::genomics::GenomicsLink::new(),
        }
    }
}

impl DomainEncoder for GenomicsEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "kmer" => {
                // JSON: "ATGC" or {"sequence": "ATGC"}
                let seq_str = json
                    .as_str()
                    .or_else(|| json.get("sequence").and_then(|v| v.as_str()))
                    .ok_or("kmer requires a sequence string")?;
                let bases: Vec<joule_db_domains::genomics::Base> = seq_str
                    .chars()
                    .map(joule_db_domains::genomics::Base::from_char)
                    .collect();
                Ok(EncoderResult::Vector(self.link.encode_kmer(&bases)))
            }
            "sequence" => {
                // JSON: {"id": "seq1", "sequence": "ATGCGATC"}
                let id = json.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                let seq_str = json
                    .get("sequence")
                    .and_then(|v| v.as_str())
                    .ok_or("sequence requires a 'sequence' field")?;
                let dna = joule_db_domains::genomics::DnaSequence::from_string(id, seq_str);
                Ok(EncoderResult::Vector(self.link.encode_sequence(&dna)))
            }
            "variant" => {
                let v: joule_db_domains::genomics::Variant = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_variant(&v)))
            }
            "gene" => {
                let g: joule_db_domains::genomics::Gene = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_gene(&g)))
            }
            "protein" => {
                // ProteinSequence doesn't derive Deserialize — custom parse
                let id = json.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                let sequence = json
                    .get("sequence")
                    .and_then(|v| v.as_str())
                    .ok_or("protein requires a 'sequence' field")?;
                let gene_id = json
                    .get("gene_id")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let protein = joule_db_domains::genomics::ProteinSequence {
                    id: id.to_string(),
                    sequence: sequence.to_string(),
                    gene_id,
                };
                Ok(EncoderResult::Vector(self.link.encode_protein(&protein)))
            }
            _ => Err(format!("Unknown genomics method: '{}'", method)),
        }
    }
}

// ============================================================================
// Legal Domain
// ============================================================================

struct LegalEncoder {
    link: joule_db_domains::legal::LegalLink,
}

impl LegalEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::legal::LegalLink::new(),
        }
    }
}

impl DomainEncoder for LegalEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "document" => {
                let d: joule_db_domains::legal::LegalDocument = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_document(&d)))
            }
            "clause" => {
                let c: joule_db_domains::legal::Clause = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_clause(&c)))
            }
            "regulation" => {
                let r: joule_db_domains::legal::Regulation = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_regulation(&r)))
            }
            "case_law" => {
                let c: joule_db_domains::legal::CaseLaw = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_case_law(&c)))
            }
            _ => Err(format!("Unknown legal method: '{}'", method)),
        }
    }
}

// ============================================================================
// Energy Domain
// ============================================================================

struct EnergyEncoder {
    link: joule_db_domains::energy::EnergyLink,
}

impl EnergyEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::energy::EnergyLink::new(),
        }
    }
}

impl DomainEncoder for EnergyEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "power_plant" => {
                let p: joule_db_domains::energy::PowerPlant = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_power_plant(&p)))
            }
            "grid_node" => {
                let g: joule_db_domains::energy::GridNode = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_grid_node(&g)))
            }
            "meter_reading" => {
                // Special: takes (MeterReading, ConsumerType) — JSON envelope
                let reading: joule_db_domains::energy::MeterReading =
                    if let Some(r) = json.get("reading") {
                        deser(r)?
                    } else {
                        deser(json)?
                    };
                let consumer_type: joule_db_domains::energy::ConsumerType =
                    if let Some(ct) = json.get("consumer_type") {
                        deser(ct)?
                    } else {
                        joule_db_domains::energy::ConsumerType::Residential
                    };
                Ok(EncoderResult::Vector(
                    self.link.encode_meter_reading(&reading, consumer_type),
                ))
            }
            "smart_meter" => {
                let m: joule_db_domains::energy::SmartMeter = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_smart_meter(&m)))
            }
            _ => Err(format!("Unknown energy method: '{}'", method)),
        }
    }
}

// ============================================================================
// AdTech Domain
// ============================================================================

struct AdTechEncoder {
    link: joule_db_domains::adtech::AdTechLink,
}

impl AdTechEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::adtech::AdTechLink::new(),
        }
    }
}

impl DomainEncoder for AdTechEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "user" => {
                let u: joule_db_domains::adtech::UserProfile = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_user(&u)))
            }
            "campaign" => {
                let c: joule_db_domains::adtech::Campaign = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_campaign(&c)))
            }
            "bid_request" => {
                let b: joule_db_domains::adtech::BidRequest = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_bid_request(&b)))
            }
            _ => Err(format!("Unknown adtech method: '{}'", method)),
        }
    }
}

// ============================================================================
// Agri Domain
// ============================================================================

struct AgriEncoder {
    link: joule_db_domains::agri::AgriLink,
}

impl AgriEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::agri::AgriLink::new(),
        }
    }
}

impl DomainEncoder for AgriEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "field" => {
                let f: joule_db_domains::agri::Field = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_field(&f)))
            }
            "observation" => {
                let o: joule_db_domains::agri::CropObservation = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_observation(&o)))
            }
            "weather" => {
                let w: joule_db_domains::agri::WeatherData = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_weather(&w)))
            }
            _ => Err(format!("Unknown agri method: '{}'", method)),
        }
    }
}

// ============================================================================
// Graph Domain
// ============================================================================

struct GraphEncoder {
    link: joule_db_domains::graph::GraphLink,
}

impl GraphEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::graph::GraphLink::new(),
        }
    }
}

impl DomainEncoder for GraphEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "entity" => {
                let e: joule_db_domains::graph::Entity = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_entity(&e)))
            }
            "relationship" => {
                let r: joule_db_domains::graph::Relationship = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_relationship(&r)))
            }
            "triple" => {
                let t: joule_db_domains::graph::Triple = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_triple(&t)))
            }
            _ => Err(format!("Unknown graph method: '{}'", method)),
        }
    }
}

// ============================================================================
// Edu Domain
// ============================================================================

struct EduEncoder {
    link: joule_db_domains::edu::EduLink,
}

impl EduEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::edu::EduLink::new(),
        }
    }
}

impl DomainEncoder for EduEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "student" => {
                let s: joule_db_domains::edu::Student = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_student(&s)))
            }
            "course" => {
                let c: joule_db_domains::edu::Course = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_course(&c)))
            }
            "assignment" => {
                let a: joule_db_domains::edu::Assignment = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_assignment(&a)))
            }
            _ => Err(format!("Unknown edu method: '{}'", method)),
        }
    }
}

// ============================================================================
// Supply Domain
// ============================================================================

struct SupplyEncoder {
    link: joule_db_domains::supply::SupplyLink,
}

impl SupplyEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::supply::SupplyLink::new(),
        }
    }
}

impl DomainEncoder for SupplyEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "shipment" => {
                let s: joule_db_domains::supply::Shipment = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_shipment(&s)))
            }
            "supplier" => {
                let s: joule_db_domains::supply::Supplier = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_supplier(&s)))
            }
            "warehouse" => {
                let w: joule_db_domains::supply::Warehouse = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_warehouse(&w)))
            }
            "route" => {
                let r: joule_db_domains::supply::Route = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_route(&r)))
            }
            _ => Err(format!("Unknown supply method: '{}'", method)),
        }
    }
}

// ============================================================================
// Retail Domain
// ============================================================================

struct RetailEncoder {
    link: joule_db_domains::retail::RetailLink,
}

impl RetailEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::retail::RetailLink::new(),
        }
    }
}

impl DomainEncoder for RetailEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "product" => {
                let p: joule_db_domains::retail::Product = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_product(&p)))
            }
            "customer" => {
                let c: joule_db_domains::retail::Customer = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_customer(&c)))
            }
            "transaction" => {
                let t: joule_db_domains::retail::Transaction = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_transaction(&t)))
            }
            "cart_session" => {
                let c: joule_db_domains::retail::CartSession = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_cart_session(&c)))
            }
            _ => Err(format!("Unknown retail method: '{}'", method)),
        }
    }
}

// ============================================================================
// Auto Domain
// ============================================================================

struct AutoEncoder {
    link: joule_db_domains::auto::AutoLink,
}

impl AutoEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::auto::AutoLink::new(),
        }
    }
}

impl DomainEncoder for AutoEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "object" => {
                let o: joule_db_domains::auto::DetectedObject = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_object(&o)))
            }
            "scene" => {
                let objects: Vec<joule_db_domains::auto::DetectedObject> = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_scene(&objects)))
            }
            "landmark" => {
                let l: joule_db_domains::auto::Landmark = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_landmark(&l)))
            }
            "agent" => {
                let a: joule_db_domains::auto::AgentState = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_agent(&a)))
            }
            _ => Err(format!("Unknown auto method: '{}'", method)),
        }
    }
}

// ============================================================================
// Gaming Domain
// ============================================================================

struct GamingEncoder {
    link: joule_db_domains::gaming::GamingLink,
}

impl GamingEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::gaming::GamingLink::new(),
        }
    }
}

impl DomainEncoder for GamingEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "player" => {
                let p: joule_db_domains::gaming::Player = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_player(&p)))
            }
            "session" => {
                let s: joule_db_domains::gaming::GameSession = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_session(&s)))
            }
            "item" => {
                let i: joule_db_domains::gaming::GameItem = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_item(&i)))
            }
            "achievement" => {
                let a: joule_db_domains::gaming::Achievement = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_achievement(&a)))
            }
            _ => Err(format!("Unknown gaming method: '{}'", method)),
        }
    }
}

// ============================================================================
// Insurance Domain
// ============================================================================

struct InsuranceEncoder {
    link: joule_db_domains::insurance::InsuranceLink,
}

impl InsuranceEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::insurance::InsuranceLink::new(),
        }
    }
}

impl DomainEncoder for InsuranceEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "policy" => {
                let p: joule_db_domains::insurance::Policy = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_policy(&p)))
            }
            "claim" => {
                let c: joule_db_domains::insurance::Claim = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_claim(&c)))
            }
            "risk_profile" => {
                let r: joule_db_domains::insurance::RiskProfile = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_risk_profile(&r)))
            }
            _ => Err(format!("Unknown insurance method: '{}'", method)),
        }
    }
}

// ============================================================================
// Spatial Domain
// ============================================================================

struct SpatialEncoder {
    link: joule_db_domains::spatial::SpatialLink,
}

impl SpatialEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::spatial::SpatialLink::new(),
        }
    }
}

impl DomainEncoder for SpatialEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "location" => {
                let l: joule_db_domains::spatial::GeoPoint = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_location(&l)))
            }
            "poi" => {
                let p: joule_db_domains::spatial::PointOfInterest = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_poi(&p)))
            }
            "geofence" => {
                let g: joule_db_domains::spatial::Geofence = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_geofence(&g)))
            }
            "trajectory" => {
                let t: joule_db_domains::spatial::Trajectory = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_trajectory(&t)))
            }
            "event" => {
                let e: joule_db_domains::spatial::SpatialEvent = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_event(&e)))
            }
            _ => Err(format!("Unknown spatial method: '{}'", method)),
        }
    }
}

// ============================================================================
// Multimodal Domain
// ============================================================================

struct MultimodalEncoder {
    link: joule_db_domains::multimodal::MultimodalLink,
}

impl MultimodalEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::multimodal::MultimodalLink::new(),
        }
    }
}

impl DomainEncoder for MultimodalEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "text" => {
                let t: joule_db_domains::multimodal::TextData = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_text(&t)))
            }
            "image" => {
                let i: joule_db_domains::multimodal::ImageData = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_image(&i)))
            }
            "audio" => {
                let a: joule_db_domains::multimodal::AudioData = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_audio(&a)))
            }
            "multimodal" => {
                let m: joule_db_domains::multimodal::MultimodalEntity = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_multimodal(&m)))
            }
            _ => Err(format!("Unknown multimodal method: '{}'", method)),
        }
    }
}

// ============================================================================
// Telecom Domain
// ============================================================================

struct TelecomEncoder {
    link: joule_db_domains::telecom::TelecomLink,
}

impl TelecomEncoder {
    fn new() -> Self {
        Self {
            link: joule_db_domains::telecom::TelecomLink::new(),
        }
    }
}

impl DomainEncoder for TelecomEncoder {
    fn encode(&mut self, method: &str, json: &serde_json::Value) -> Result<EncoderResult, String> {
        match method {
            "cell_tower" => {
                let c: joule_db_domains::telecom::CellTower = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_cell_tower(&c)))
            }
            "network_session" => {
                let s: joule_db_domains::telecom::NetworkSession = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_network_session(&s)))
            }
            "subscriber" => {
                let s: joule_db_domains::telecom::Subscriber = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_subscriber(&s)))
            }
            "alarm" => {
                let a: joule_db_domains::telecom::NetworkAlarm = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_alarm(&a)))
            }
            "quality_metrics" => {
                let q: joule_db_domains::telecom::QualityMetrics = deser(json)?;
                Ok(EncoderResult::Vector(self.link.encode_quality_metrics(&q)))
            }
            _ => Err(format!("Unknown telecom method: '{}'", method)),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let hv = BinaryHV::random(DEFAULT_DIMENSION, 42);
        let text = serialize_hv(&hv);
        assert!(text.starts_with("hdc:10000:"));
        let restored = deserialize_hv(&text).expect("deserialize failed");
        assert_eq!(restored.dimension(), hv.dimension());
        assert_eq!(restored.as_words(), hv.as_words());
    }

    #[test]
    fn test_serialize_small_dimension() {
        let hv = BinaryHV::random(64, 123);
        let text = serialize_hv(&hv);
        assert!(text.starts_with("hdc:64:"));
        let restored = deserialize_hv(&text).unwrap();
        assert_eq!(restored.dimension(), 64);
        assert_eq!(restored.hamming_distance(&hv), 0);
    }

    #[test]
    fn test_deserialize_invalid() {
        assert!(deserialize_hv("garbage").is_none());
        assert!(deserialize_hv("hdc:abc:data").is_none());
        assert!(deserialize_hv("notHdc:10000:AAAA").is_none());
        assert!(deserialize_hv("hdc:10000:!!!invalid-base64!!!").is_none());
    }

    #[test]
    fn test_deserialize_dimension_mismatch() {
        let hv = BinaryHV::random(64, 42);
        let text = serialize_hv(&hv);
        // Change the dimension in the header to something wrong
        let wrong = text.replacen("64", "128", 1);
        assert!(deserialize_hv(&wrong).is_none());
    }

    #[test]
    fn test_hdc_encode_market_trade() {
        let json = r#"{"symbol":"AAPL","price":150.0,"quantity":100.0,"side":"Buy"}"#;
        let result = hdc_encode("market", "trade", json);
        assert!(result.is_ok());
        let hv_text = result.unwrap();
        assert!(hv_text.starts_with("hdc:10000:"));
        // Deserialize to verify it's a valid BinaryHV
        let hv = deserialize_hv(&hv_text).expect("should deserialize");
        assert_eq!(hv.dimension(), DEFAULT_DIMENSION);
    }

    #[test]
    fn test_hdc_encode_health_patient() {
        let json = r#"{"patient_id":"P001","age":45,"sex":"Male","blood_type":"OPositive","weight_kg":80.0,"height_cm":175.0}"#;
        let result = hdc_encode("health", "patient", json);
        assert!(result.is_ok());
        let hv_text = result.unwrap();
        assert!(hv_text.starts_with("hdc:10000:"));
    }

    #[test]
    fn test_hdc_encode_deterministic() {
        let json = r#"{"symbol":"MSFT","price":300.0,"quantity":50.0,"side":"Sell"}"#;
        let r1 = hdc_encode("market", "trade", json).unwrap();
        let r2 = hdc_encode("market", "trade", json).unwrap();
        assert_eq!(r1, r2, "Same input should produce same output");
    }

    #[test]
    fn test_hdc_encode_unknown_domain() {
        let result = hdc_encode("fantasy", "spell", "{}");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown HDC domain"));
    }

    #[test]
    fn test_hdc_encode_unknown_method() {
        let result = hdc_encode("market", "nonexistent", "{}");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown market method"));
    }

    #[test]
    fn test_hdc_encode_invalid_json() {
        let result = hdc_encode("market", "trade", "not-json");
        assert!(result.is_err());
    }

    #[test]
    fn test_hdc_similarity_identical() {
        let hv = BinaryHV::random(DEFAULT_DIMENSION, 42);
        let text = serialize_hv(&hv);
        let sim = hv.similarity(&hv);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hdc_distance_identical() {
        let hv = BinaryHV::random(DEFAULT_DIMENSION, 42);
        assert_eq!(hv.hamming_distance(&hv), 0);
    }

    #[test]
    fn test_hdc_bind_self_inverse() {
        let a = BinaryHV::random(DEFAULT_DIMENSION, 1);
        let b = BinaryHV::random(DEFAULT_DIMENSION, 2);
        let bound = a.bind(&b);
        let restored = bound.bind(&b);
        assert_eq!(restored.hamming_distance(&a), 0, "XOR bind is self-inverse");
    }

    #[test]
    fn test_hdc_bundle() {
        let a = BinaryHV::random(DEFAULT_DIMENSION, 1);
        let b = BinaryHV::random(DEFAULT_DIMENSION, 2);
        let c = BinaryHV::random(DEFAULT_DIMENSION, 3);
        let mut acc = BundleAccumulator::new(DEFAULT_DIMENSION);
        acc.add(&a);
        acc.add(&b);
        acc.add(&c);
        let bundled = acc.threshold();
        // Bundled vector should be similar to each component (>0.5)
        assert!(bundled.similarity(&a) > 0.5);
        assert!(bundled.similarity(&b) > 0.5);
        assert!(bundled.similarity(&c) > 0.5);
    }

    #[test]
    fn test_hdc_encode_text_fuzzy_similarity() {
        let hv1 = BinaryHV::from_bytes("alice".as_bytes(), DEFAULT_DIMENSION);
        let hv2 = BinaryHV::from_bytes("alcie".as_bytes(), DEFAULT_DIMENSION);
        let hv3 = BinaryHV::from_bytes("bob".as_bytes(), DEFAULT_DIMENSION);
        // "alice" and "alcie" should be more similar than "alice" and "bob"
        assert!(hv1.similarity(&hv2) > hv1.similarity(&hv3));
    }

    #[test]
    fn test_hdc_encode_hash() {
        let hv = BinaryHV::from_data("test data".as_bytes(), DEFAULT_DIMENSION);
        assert_eq!(hv.dimension(), DEFAULT_DIMENSION);
        // Hash of same data should be identical
        let hv2 = BinaryHV::from_data("test data".as_bytes(), DEFAULT_DIMENSION);
        assert_eq!(hv.hamming_distance(&hv2), 0);
    }

    #[test]
    fn test_hdc_dims() {
        let hv = BinaryHV::random(512, 42);
        let text = serialize_hv(&hv);
        let restored = deserialize_hv(&text).unwrap();
        assert_eq!(restored.dimension(), 512);
    }

    #[test]
    fn test_genomics_kmer() {
        let result = hdc_encode("genomics", "kmer", r#""ATGCGATC""#);
        assert!(result.is_ok());
        assert!(result.unwrap().starts_with("hdc:10000:"));
    }

    #[test]
    fn test_genomics_sequence() {
        let json = r#"{"id":"seq1","sequence":"ATGCGATCAATGC"}"#;
        let result = hdc_encode("genomics", "sequence", json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_trend() {
        let json = r#"[{"timestamp":1,"value":1.0,"labels":{}},{"timestamp":2,"value":2.0,"labels":{}},{"timestamp":3,"value":3.0,"labels":{}}]"#;
        let result = hdc_encode("temporal", "trend", json);
        assert!(result.is_ok());
        let text = result.unwrap();
        // trend returns a string like "Up", not an HDC vector
        assert!(
            text == "Up" || text == "Down" || text == "Flat" || text == "Volatile",
            "Got: {}",
            text
        );
    }

    #[test]
    fn test_energy_meter_reading_envelope() {
        let json = r#"{
            "reading": {"timestamp":1000,"consumption_kwh":42.5,"voltage":230.0,"power_factor":0.95,"time_of_use":"Peak"},
            "consumer_type": "Residential"
        }"#;
        let result = hdc_encode("energy", "meter_reading", json);
        assert!(result.is_ok());
    }
}
