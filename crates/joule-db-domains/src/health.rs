//! HDC-powered Healthcare and Clinical Decision Support module
//!
//! This module provides holographic encoding and analysis for healthcare data:
//! - Patient records with demographics and medical history
//! - Clinical decision support with pattern-based diagnosis
//! - Drug interaction checking via holographic encoding
//! - Risk prediction for patient outcomes
//! - Similar patient cohort discovery

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Healthcare Types
// ============================================================================

/// Patient demographics and identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patient {
    pub patient_id: String,
    pub age: u8,
    pub sex: Sex,
    pub blood_type: BloodType,
    pub weight_kg: f32,
    pub height_cm: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BloodType {
    APositive,
    ANegative,
    BPositive,
    BNegative,
    ABPositive,
    ABNegative,
    OPositive,
    ONegative,
}

/// Vital signs measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vitals {
    pub timestamp: u64,
    pub heart_rate: u16,
    pub systolic_bp: u16,
    pub diastolic_bp: u16,
    pub temperature: f32,
    pub respiratory_rate: u16,
    pub oxygen_saturation: u8,
}

/// Laboratory test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabResult {
    pub test_code: String,
    pub test_name: String,
    pub value: f64,
    pub unit: String,
    pub reference_low: f64,
    pub reference_high: f64,
    pub timestamp: u64,
}

impl LabResult {
    pub fn is_normal(&self) -> bool {
        self.value >= self.reference_low && self.value <= self.reference_high
    }

    pub fn is_critical(&self) -> bool {
        self.value < self.reference_low * 0.5 || self.value > self.reference_high * 2.0
    }
}

/// ICD-10 diagnosis code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    pub icd_code: String,
    pub description: String,
    pub diagnosed_date: u64,
    pub is_primary: bool,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity {
    Mild,
    Moderate,
    Severe,
    Critical,
}

/// Medication prescription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Medication {
    pub rxcui: String,
    pub name: String,
    pub dose: f32,
    pub unit: String,
    pub frequency: MedFrequency,
    pub route: AdministrationRoute,
    pub start_date: u64,
    pub end_date: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MedFrequency {
    Once,
    Daily,
    BID,
    TID,
    QID,
    PRN,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdministrationRoute {
    Oral,
    IV,
    IM,
    SC,
    Topical,
    Inhaled,
    Rectal,
    Sublingual,
}

/// Medical procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub cpt_code: String,
    pub description: String,
    pub performed_date: u64,
    pub outcome: ProcedureOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcedureOutcome {
    Successful,
    PartialSuccess,
    Unsuccessful,
    Complications,
}

/// Clinical symptom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symptom {
    pub name: String,
    pub body_system: BodySystem,
    pub severity: Severity,
    pub onset_date: u64,
    pub duration_days: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodySystem {
    Cardiovascular,
    Respiratory,
    Gastrointestinal,
    Neurological,
    Musculoskeletal,
    Integumentary,
    Endocrine,
    Renal,
    Hematologic,
    Immunologic,
    Psychiatric,
    General,
}

/// Complete medical encounter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalEncounter {
    pub encounter_id: String,
    pub patient_id: String,
    pub encounter_date: u64,
    pub encounter_type: EncounterType,
    pub vitals: Option<Vitals>,
    pub symptoms: Vec<Symptom>,
    pub diagnoses: Vec<Diagnosis>,
    pub medications: Vec<Medication>,
    pub procedures: Vec<Procedure>,
    pub lab_results: Vec<LabResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncounterType {
    Emergency,
    Inpatient,
    Outpatient,
    Telemedicine,
    Preventive,
}

// ============================================================================
// Holographic Healthcare Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder for healthcare domain data
    pub struct HealthLink {
        seed: 0x0EA1_7000,
        dimension: 10000,
        fields: [
            "patient", "age", "sex", "blood_type", "weight", "height",
            "heart_rate", "systolic", "diastolic", "temperature", "resp_rate", "spo2",
            "diagnosis", "icd_code", "severity", "primary",
            "medication", "rxcui", "dose", "route", "frequency",
            "lab", "loinc", "value", "reference",
            "symptom", "body_system", "onset", "duration",
            "procedure", "cpt", "outcome",
            "encounter", "encounter_type", "timestamp"
        ],
        scalars: ["age", "weight", "height", "heart_rate", "bp", "temperature",
                   "resp_rate", "spo2", "lab_value", "dose", "duration"],
        enums: {
            sex_vectors: Sex => [Sex::Male, Sex::Female, Sex::Other],
            blood_type_vectors: BloodType => [BloodType::APositive, BloodType::ANegative, BloodType::BPositive, BloodType::BNegative, BloodType::ABPositive, BloodType::ABNegative, BloodType::OPositive, BloodType::ONegative],
            severity_vectors: Severity => [Severity::Mild, Severity::Moderate, Severity::Severe, Severity::Critical],
            body_system_vectors: BodySystem => [BodySystem::Cardiovascular, BodySystem::Respiratory, BodySystem::Gastrointestinal, BodySystem::Neurological, BodySystem::Musculoskeletal, BodySystem::Integumentary, BodySystem::Endocrine, BodySystem::Renal, BodySystem::Hematologic, BodySystem::Immunologic, BodySystem::Psychiatric, BodySystem::General],
            encounter_type_vectors: EncounterType => [EncounterType::Emergency, EncounterType::Inpatient, EncounterType::Outpatient, EncounterType::Telemedicine, EncounterType::Preventive],
            route_vectors: AdministrationRoute => [AdministrationRoute::Oral, AdministrationRoute::IV, AdministrationRoute::IM, AdministrationRoute::SC, AdministrationRoute::Topical, AdministrationRoute::Inhaled, AdministrationRoute::Rectal, AdministrationRoute::Sublingual],
            frequency_vectors: MedFrequency => [MedFrequency::Once, MedFrequency::Daily, MedFrequency::BID, MedFrequency::TID, MedFrequency::QID, MedFrequency::PRN, MedFrequency::Weekly, MedFrequency::Monthly],
            outcome_vectors: ProcedureOutcome => [ProcedureOutcome::Successful, ProcedureOutcome::PartialSuccess, ProcedureOutcome::Unsuccessful, ProcedureOutcome::Complications]
        },
        dynamic: {
            icd_vectors: "icd",
            rxcui_vectors: "rxcui",
            loinc_vectors: "loinc"
        },
    }
}

impl HealthLink {
    pub fn encode_patient(&self, patient: &Patient) -> BinaryHV {
        let sex_hv = self.field_vectors["sex"].bind(&self.sex_vectors[&patient.sex]);
        let blood_hv =
            self.field_vectors["blood_type"].bind(&self.blood_type_vectors[&patient.blood_type]);
        let age_hv =
            self.field_vectors["age"].bind(&self.encode_scalar("age", patient.age as u32, 120));
        let weight_hv = self.field_vectors["weight"].bind(&self.encode_scalar(
            "weight",
            patient.weight_kg as u32,
            300,
        ));
        let height_hv = self.field_vectors["height"].bind(&self.encode_scalar(
            "height",
            patient.height_cm as u32,
            250,
        ));

        let patient_core = self.bundle(&[sex_hv, blood_hv, age_hv, weight_hv, height_hv]);
        self.field_vectors["patient"].bind(&patient_core)
    }

    pub fn encode_vitals(&self, vitals: &Vitals) -> BinaryHV {
        let hr_hv = self.field_vectors["heart_rate"].bind(&self.encode_scalar(
            "heart_rate",
            vitals.heart_rate as u32,
            220,
        ));
        let sys_hv = self.field_vectors["systolic"].bind(&self.encode_scalar(
            "bp",
            vitals.systolic_bp as u32,
            250,
        ));
        let dia_hv = self.field_vectors["diastolic"].bind(&self.encode_scalar(
            "bp",
            vitals.diastolic_bp as u32,
            150,
        ));
        let temp_scaled = ((vitals.temperature - 30.0) * 10.0) as u32;
        let temp_hv = self.field_vectors["temperature"].bind(&self.encode_scalar(
            "temperature",
            temp_scaled,
            150,
        ));
        let rr_hv = self.field_vectors["resp_rate"].bind(&self.encode_scalar(
            "resp_rate",
            vitals.respiratory_rate as u32,
            60,
        ));
        let spo2_hv = self.field_vectors["spo2"].bind(&self.encode_scalar(
            "spo2",
            vitals.oxygen_saturation as u32,
            100,
        ));

        self.bundle(&[hr_hv, sys_hv, dia_hv, temp_hv, rr_hv, spo2_hv])
    }

    pub fn encode_diagnosis(&mut self, diagnosis: &Diagnosis) -> BinaryHV {
        let icd_vec = self.icd_vectors(&diagnosis.icd_code);
        let icd_hv = self.field_vectors["icd_code"].bind(&icd_vec);
        let sev_hv =
            self.field_vectors["severity"].bind(&self.severity_vectors[&diagnosis.severity]);

        let mut components = vec![icd_hv, sev_hv];
        if diagnosis.is_primary {
            let primary_hv = self.field_vectors["primary"].permute(1);
            components.push(primary_hv);
        }

        let diag_core = self.bundle(&components);
        self.field_vectors["diagnosis"].bind(&diag_core)
    }

    pub fn encode_medication(&mut self, medication: &Medication) -> BinaryHV {
        let rx_vec = self.rxcui_vectors(&medication.rxcui);
        let rx_hv = self.field_vectors["rxcui"].bind(&rx_vec);
        let route_hv = self.field_vectors["route"].bind(&self.route_vectors[&medication.route]);
        let freq_hv =
            self.field_vectors["frequency"].bind(&self.frequency_vectors[&medication.frequency]);
        let dose_hv = self.field_vectors["dose"].bind(&self.encode_scalar(
            "dose",
            medication.dose as u32,
            1000,
        ));

        let med_core = self.bundle(&[rx_hv, route_hv, freq_hv, dose_hv]);
        self.field_vectors["medication"].bind(&med_core)
    }

    pub fn encode_lab_result(&mut self, lab: &LabResult) -> BinaryHV {
        let loinc_vec = self.loinc_vectors(&lab.test_code);
        let loinc_hv = self.field_vectors["loinc"].bind(&loinc_vec);

        let range = lab.reference_high - lab.reference_low;
        let normalized = if range > 0.0 {
            ((lab.value - lab.reference_low) / range * 100.0).clamp(0.0, 200.0) as u32
        } else {
            100
        };
        let val_hv =
            self.field_vectors["value"].bind(&self.encode_scalar("lab_value", normalized, 200));

        let lab_core = self.bundle(&[loinc_hv, val_hv]);
        self.field_vectors["lab"].bind(&lab_core)
    }

    pub fn encode_symptom(&self, symptom: &Symptom) -> BinaryHV {
        let system_hv =
            self.field_vectors["body_system"].bind(&self.body_system_vectors[&symptom.body_system]);
        let sev_hv = self.field_vectors["severity"].bind(&self.severity_vectors[&symptom.severity]);

        let mut components = vec![system_hv, sev_hv];
        if let Some(duration) = symptom.duration_days {
            let dur_hv = self.field_vectors["duration"].bind(&self.encode_scalar(
                "duration",
                duration.min(365),
                365,
            ));
            components.push(dur_hv);
        }

        let sym_core = self.bundle(&components);
        self.field_vectors["symptom"].bind(&sym_core)
    }

    pub fn encode_procedure(&self, procedure: &Procedure) -> BinaryHV {
        let cpt_seed = procedure
            .cpt_code
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let cpt_hv = self.field_vectors["cpt"].bind(&BinaryHV::random(DIMENSION, cpt_seed));
        let outcome_hv =
            self.field_vectors["outcome"].bind(&self.outcome_vectors[&procedure.outcome]);

        let proc_core = self.bundle(&[cpt_hv, outcome_hv]);
        self.field_vectors["procedure"].bind(&proc_core)
    }

    pub fn encode_encounter(&mut self, encounter: &MedicalEncounter) -> BinaryHV {
        let mut components = Vec::new();

        let type_hv = self.field_vectors["encounter_type"]
            .bind(&self.encounter_type_vectors[&encounter.encounter_type]);
        components.push(type_hv);

        if let Some(ref vitals) = encounter.vitals {
            components.push(self.encode_vitals(vitals));
        }

        for symptom in &encounter.symptoms {
            components.push(self.encode_symptom(symptom));
        }

        for diagnosis in encounter.diagnoses.clone() {
            components.push(self.encode_diagnosis(&diagnosis));
        }

        for medication in encounter.medications.clone() {
            components.push(self.encode_medication(&medication));
        }

        for procedure in &encounter.procedures {
            components.push(self.encode_procedure(procedure));
        }

        for lab in encounter.lab_results.clone() {
            components.push(self.encode_lab_result(&lab));
        }

        let enc_core = self.bundle(&components);
        self.field_vectors["encounter"].bind(&enc_core)
    }
}

// ============================================================================
// Patient Database - Similar Patient Discovery
// ============================================================================

pub struct PatientDatabase {
    encoder: HealthLink,
    patient_hologram: BundleAccumulator,
    patient_vectors: HashMap<String, BinaryHV>,
    patient_records: HashMap<String, PatientRecord>,
}

#[derive(Debug, Clone)]
pub struct PatientRecord {
    pub patient: Patient,
    pub diagnoses: Vec<Diagnosis>,
    pub medications: Vec<Medication>,
    pub encounters: Vec<MedicalEncounter>,
}

impl PatientDatabase {
    pub fn new() -> Self {
        Self {
            encoder: HealthLink::new(),
            patient_hologram: BundleAccumulator::new(DIMENSION),
            patient_vectors: HashMap::new(),
            patient_records: HashMap::new(),
        }
    }

    pub fn add_patient(&mut self, record: PatientRecord) {
        let mut components = vec![self.encoder.encode_patient(&record.patient)];

        for diag in record.diagnoses.clone() {
            components.push(self.encoder.encode_diagnosis(&diag));
        }

        for med in record.medications.clone() {
            components.push(self.encoder.encode_medication(&med));
        }

        let patient_hv = self.encoder.bundle(&components);

        self.patient_hologram.add(&patient_hv);
        self.patient_vectors
            .insert(record.patient.patient_id.clone(), patient_hv);
        self.patient_records
            .insert(record.patient.patient_id.clone(), record);
    }

    pub fn find_similar(
        &self,
        patient_id: &str,
        min_similarity: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query_hv = match self.patient_vectors.get(patient_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut results: Vec<(String, f32)> = self
            .patient_vectors
            .iter()
            .filter(|(id, _)| *id != patient_id)
            .map(|(id, hv)| {
                let sim = query_hv.similarity(hv);
                (id.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= min_similarity)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn find_by_diagnosis_pattern(
        &mut self,
        diagnoses: &[Diagnosis],
        min_similarity: f32,
    ) -> Vec<(String, f32)> {
        let mut diag_components = Vec::new();
        for diag in diagnoses {
            diag_components.push(self.encoder.encode_diagnosis(&diag.clone()));
        }
        let query_hv = self.encoder.bundle(&diag_components);

        let mut results: Vec<(String, f32)> = Vec::new();
        for (patient_id, record) in &self.patient_records {
            if record.diagnoses.is_empty() {
                continue;
            }

            let mut encoder_clone = HealthLink::new();
            let patient_diag_components: Vec<BinaryHV> = record
                .diagnoses
                .iter()
                .map(|d| encoder_clone.encode_diagnosis(&d.clone()))
                .collect();
            let patient_diag_hv = encoder_clone.bundle(&patient_diag_components);

            let sim = query_hv.similarity(&patient_diag_hv);
            if sim >= min_similarity {
                results.push((patient_id.clone(), sim));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn patient_count(&self) -> usize {
        self.patient_records.len()
    }
}

impl Default for PatientDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Clinical Decision Support - Diagnosis Suggestion
// ============================================================================

pub struct ClinicalDecisionSupport {
    encoder: HealthLink,
    symptom_diagnosis_patterns: HashMap<String, BundleAccumulator>,
    diagnosis_vectors: HashMap<String, BinaryHV>,
    lab_diagnosis_patterns: HashMap<String, BundleAccumulator>,
    case_hologram: BundleAccumulator,
    case_outcomes: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct DiagnosisSuggestion {
    pub icd_code: String,
    pub description: String,
    pub confidence: f32,
    pub supporting_symptoms: Vec<String>,
    pub supporting_labs: Vec<String>,
}

impl ClinicalDecisionSupport {
    pub fn new() -> Self {
        Self {
            encoder: HealthLink::new(),
            symptom_diagnosis_patterns: HashMap::new(),
            diagnosis_vectors: HashMap::new(),
            lab_diagnosis_patterns: HashMap::new(),
            case_hologram: BundleAccumulator::new(DIMENSION),
            case_outcomes: HashMap::new(),
        }
    }

    pub fn learn_case(
        &mut self,
        case_id: &str,
        symptoms: &[Symptom],
        labs: &[LabResult],
        diagnoses: &[Diagnosis],
    ) {
        let symptom_components: Vec<BinaryHV> = symptoms
            .iter()
            .map(|s| self.encoder.encode_symptom(s))
            .collect();

        if !symptom_components.is_empty() {
            let symptom_hv = self.encoder.bundle(&symptom_components);

            for diag in diagnoses {
                let pattern = self
                    .symptom_diagnosis_patterns
                    .entry(diag.icd_code.clone())
                    .or_insert_with(|| BundleAccumulator::new(DIMENSION));
                pattern.add(&symptom_hv);

                if !self.diagnosis_vectors.contains_key(&diag.icd_code) {
                    let diag_hv = self.encoder.encode_diagnosis(&diag.clone());
                    self.diagnosis_vectors
                        .insert(diag.icd_code.clone(), diag_hv);
                }
            }
        }

        let lab_components: Vec<BinaryHV> = labs
            .iter()
            .map(|l| self.encoder.encode_lab_result(&l.clone()))
            .collect();

        if !lab_components.is_empty() {
            let lab_hv = self.encoder.bundle(&lab_components);

            for diag in diagnoses {
                let pattern = self
                    .lab_diagnosis_patterns
                    .entry(diag.icd_code.clone())
                    .or_insert_with(|| BundleAccumulator::new(DIMENSION));
                pattern.add(&lab_hv);
            }
        }

        self.case_outcomes.insert(
            case_id.to_string(),
            diagnoses.iter().map(|d| d.icd_code.clone()).collect(),
        );

        let mut all_components = symptom_components;
        all_components.extend(lab_components);
        if !all_components.is_empty() {
            self.case_hologram
                .add(&self.encoder.bundle(&all_components));
        }
    }

    pub fn suggest_diagnoses(
        &mut self,
        symptoms: &[Symptom],
        labs: &[LabResult],
        limit: usize,
    ) -> Vec<DiagnosisSuggestion> {
        let mut suggestions: HashMap<String, (f32, Vec<String>, Vec<String>)> = HashMap::new();

        if !symptoms.is_empty() {
            let symptom_components: Vec<BinaryHV> = symptoms
                .iter()
                .map(|s| self.encoder.encode_symptom(s))
                .collect();
            let current_symptoms_hv = self.encoder.bundle(&symptom_components);

            for (icd_code, pattern) in &self.symptom_diagnosis_patterns {
                let pattern_hv = pattern.threshold();
                let sim = current_symptoms_hv.similarity(&pattern_hv);

                if sim > 0.3 {
                    let entry = suggestions.entry(icd_code.clone()).or_insert((
                        0.0,
                        Vec::new(),
                        Vec::new(),
                    ));
                    entry.0 += sim * 0.6;
                    entry.1 = symptoms.iter().map(|s| s.name.clone()).collect();
                }
            }
        }

        if !labs.is_empty() {
            let lab_components: Vec<BinaryHV> = labs
                .iter()
                .map(|l| self.encoder.encode_lab_result(&l.clone()))
                .collect();
            let current_labs_hv = self.encoder.bundle(&lab_components);

            for (icd_code, pattern) in &self.lab_diagnosis_patterns {
                let pattern_hv = pattern.threshold();
                let sim = current_labs_hv.similarity(&pattern_hv);

                if sim > 0.3 {
                    let entry = suggestions.entry(icd_code.clone()).or_insert((
                        0.0,
                        Vec::new(),
                        Vec::new(),
                    ));
                    entry.0 += sim * 0.4;
                    entry.2 = labs.iter().map(|l| l.test_name.clone()).collect();
                }
            }
        }

        let mut results: Vec<DiagnosisSuggestion> = suggestions
            .into_iter()
            .map(
                |(icd_code, (confidence, symptoms, labs))| DiagnosisSuggestion {
                    icd_code: icd_code.clone(),
                    description: format!("Diagnosis {}", icd_code),
                    confidence,
                    supporting_symptoms: symptoms,
                    supporting_labs: labs,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    pub fn diagnosis_count(&self) -> usize {
        self.symptom_diagnosis_patterns.len()
    }
}

impl Default for ClinicalDecisionSupport {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Drug Interaction Checker
// ============================================================================

pub struct DrugInteractionChecker {
    encoder: HealthLink,
    interaction_hologram: BundleAccumulator,
    interaction_vectors: HashMap<(String, String), BinaryHV>,
    interaction_severity: HashMap<(String, String), InteractionSeverity>,
    drug_classes: HashMap<String, HashSet<String>>,
    class_vectors: HashMap<String, BinaryHV>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionSeverity {
    Minor,
    Moderate,
    Major,
    Contraindicated,
}

#[derive(Debug, Clone)]
pub struct DrugInteraction {
    pub drug1_rxcui: String,
    pub drug2_rxcui: String,
    pub severity: InteractionSeverity,
    pub description: String,
    pub confidence: f32,
}

impl DrugInteractionChecker {
    pub fn new() -> Self {
        Self {
            encoder: HealthLink::new(),
            interaction_hologram: BundleAccumulator::new(DIMENSION),
            interaction_vectors: HashMap::new(),
            interaction_severity: HashMap::new(),
            drug_classes: HashMap::new(),
            class_vectors: HashMap::new(),
        }
    }

    pub fn register_interaction(
        &mut self,
        rxcui1: &str,
        rxcui2: &str,
        severity: InteractionSeverity,
    ) {
        let (r1, r2) = if rxcui1 < rxcui2 {
            (rxcui1.to_string(), rxcui2.to_string())
        } else {
            (rxcui2.to_string(), rxcui1.to_string())
        };

        let drug1_hv = self.encoder.rxcui_vectors(&r1);
        let drug2_hv = self.encoder.rxcui_vectors(&r2);
        let interaction_hv = drug1_hv.bind(&drug2_hv);

        self.interaction_hologram.add(&interaction_hv);
        self.interaction_vectors
            .insert((r1.clone(), r2.clone()), interaction_hv);
        self.interaction_severity.insert((r1, r2), severity);
    }

    pub fn register_drug_class(&mut self, class_name: &str, rxcui: &str) {
        self.drug_classes
            .entry(class_name.to_string())
            .or_insert_with(HashSet::new)
            .insert(rxcui.to_string());

        let drugs = self.drug_classes.get(class_name).unwrap();
        let drug_hvs: Vec<BinaryHV> = drugs
            .iter()
            .map(|rx| self.encoder.rxcui_vectors(rx))
            .collect();
        let class_hv = self.encoder.bundle(&drug_hvs);
        self.class_vectors.insert(class_name.to_string(), class_hv);
    }

    pub fn check_interactions(&self, medications: &[Medication]) -> Vec<DrugInteraction> {
        let mut interactions = Vec::new();

        for i in 0..medications.len() {
            for j in (i + 1)..medications.len() {
                let rx1 = &medications[i].rxcui;
                let rx2 = &medications[j].rxcui;

                let (r1, r2) = if rx1 < rx2 {
                    (rx1.clone(), rx2.clone())
                } else {
                    (rx2.clone(), rx1.clone())
                };

                if let Some(severity) = self.interaction_severity.get(&(r1.clone(), r2.clone())) {
                    interactions.push(DrugInteraction {
                        drug1_rxcui: rx1.clone(),
                        drug2_rxcui: rx2.clone(),
                        severity: *severity,
                        description: format!(
                            "Known {:?} interaction between {} and {}",
                            severity, medications[i].name, medications[j].name
                        ),
                        confidence: 1.0,
                    });
                    continue;
                }

                let drug1_hv = self.encoder.rxcui_vectors.get(&r1);
                let drug2_hv = self.encoder.rxcui_vectors.get(&r2);

                if let (Some(d1), Some(d2)) = (drug1_hv, drug2_hv) {
                    let potential_interaction_hv = d1.bind(d2);
                    let hologram_hv = self.interaction_hologram.threshold();
                    let sim = potential_interaction_hv.similarity(&hologram_hv);

                    if sim > 0.6 {
                        interactions.push(DrugInteraction {
                            drug1_rxcui: rx1.clone(),
                            drug2_rxcui: rx2.clone(),
                            severity: InteractionSeverity::Moderate,
                            description: format!(
                                "Potential interaction between {} and {} (similarity: {:.2})",
                                medications[i].name, medications[j].name, sim
                            ),
                            confidence: sim,
                        });
                    }
                }
            }
        }

        interactions.sort_by(|a, b| {
            let sev_ord = |s: InteractionSeverity| match s {
                InteractionSeverity::Contraindicated => 0,
                InteractionSeverity::Major => 1,
                InteractionSeverity::Moderate => 2,
                InteractionSeverity::Minor => 3,
            };
            sev_ord(a.severity).cmp(&sev_ord(b.severity))
        });

        interactions
    }

    pub fn interaction_count(&self) -> usize {
        self.interaction_severity.len()
    }
}

impl Default for DrugInteractionChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Risk Predictor - Patient Outcome Prediction
// ============================================================================

pub struct RiskPredictor {
    encoder: HealthLink,
    outcome_patterns: HashMap<RiskOutcome, BundleAccumulator>,
    outcome_counts: HashMap<RiskOutcome, usize>,
    risk_factor_weights: HashMap<String, f32>,
    training_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RiskOutcome {
    Readmission30Day,
    MortalityInHospital,
    ICUAdmission,
    ProlongedStay,
    Complication,
    NoAdverseEvent,
}

#[derive(Debug, Clone)]
pub struct RiskPrediction {
    pub outcome: RiskOutcome,
    pub probability: f32,
    pub risk_factors: Vec<String>,
}

impl RiskPredictor {
    pub fn new() -> Self {
        let mut risk_factor_weights = HashMap::new();
        risk_factor_weights.insert("age_over_65".to_string(), 1.5);
        risk_factor_weights.insert("multiple_comorbidities".to_string(), 2.0);
        risk_factor_weights.insert("abnormal_vitals".to_string(), 1.8);
        risk_factor_weights.insert("polypharmacy".to_string(), 1.3);
        risk_factor_weights.insert("recent_hospitalization".to_string(), 1.6);
        risk_factor_weights.insert("critical_lab".to_string(), 2.5);

        Self {
            encoder: HealthLink::new(),
            outcome_patterns: HashMap::new(),
            outcome_counts: HashMap::new(),
            risk_factor_weights,
            training_count: 0,
        }
    }

    pub fn train(&mut self, encounter: &MedicalEncounter, outcome: RiskOutcome) {
        let encounter_hv = self.encoder.encode_encounter(&encounter.clone());

        let pattern = self
            .outcome_patterns
            .entry(outcome)
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        pattern.add(&encounter_hv);

        *self.outcome_counts.entry(outcome).or_insert(0) += 1;
        self.training_count += 1;
    }

    pub fn predict_risk(&mut self, encounter: &MedicalEncounter) -> Vec<RiskPrediction> {
        let encounter_hv = self.encoder.encode_encounter(&encounter.clone());
        let mut predictions = Vec::new();

        for (outcome, pattern) in &self.outcome_patterns {
            let pattern_hv = pattern.threshold();
            let sim = encounter_hv.similarity(&pattern_hv);

            let mut risk_factors = Vec::new();
            let mut risk_multiplier = 1.0f32;

            if let Some(ref vitals) = encounter.vitals {
                if vitals.systolic_bp > 180 || vitals.systolic_bp < 90 {
                    risk_factors.push("abnormal_vitals".to_string());
                    risk_multiplier *= self
                        .risk_factor_weights
                        .get("abnormal_vitals")
                        .unwrap_or(&1.0);
                }
            }

            if encounter.diagnoses.len() >= 3 {
                risk_factors.push("multiple_comorbidities".to_string());
                risk_multiplier *= self
                    .risk_factor_weights
                    .get("multiple_comorbidities")
                    .unwrap_or(&1.0);
            }

            if encounter.medications.len() >= 5 {
                risk_factors.push("polypharmacy".to_string());
                risk_multiplier *= self.risk_factor_weights.get("polypharmacy").unwrap_or(&1.0);
            }

            let has_critical_lab = encounter.lab_results.iter().any(|l| l.is_critical());
            if has_critical_lab {
                risk_factors.push("critical_lab".to_string());
                risk_multiplier *= self.risk_factor_weights.get("critical_lab").unwrap_or(&1.0);
            }

            let base_probability = sim;
            let adjusted_probability = (base_probability * risk_multiplier).min(0.99);

            if adjusted_probability > 0.1 {
                predictions.push(RiskPrediction {
                    outcome: *outcome,
                    probability: adjusted_probability,
                    risk_factors,
                });
            }
        }

        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions
    }

    pub fn training_count(&self) -> usize {
        self.training_count
    }
}

impl Default for RiskPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_patient() -> Patient {
        Patient {
            patient_id: "P001".to_string(),
            age: 65,
            sex: Sex::Male,
            blood_type: BloodType::APositive,
            weight_kg: 80.0,
            height_cm: 175.0,
        }
    }

    fn create_test_vitals() -> Vitals {
        Vitals {
            timestamp: 1000,
            heart_rate: 72,
            systolic_bp: 120,
            diastolic_bp: 80,
            temperature: 37.0,
            respiratory_rate: 16,
            oxygen_saturation: 98,
        }
    }

    fn create_test_diagnosis() -> Diagnosis {
        Diagnosis {
            icd_code: "I10".to_string(),
            description: "Essential hypertension".to_string(),
            diagnosed_date: 1000,
            is_primary: true,
            severity: Severity::Moderate,
        }
    }

    fn create_test_medication() -> Medication {
        Medication {
            rxcui: "197361".to_string(),
            name: "Lisinopril".to_string(),
            dose: 10.0,
            unit: "mg".to_string(),
            frequency: MedFrequency::Daily,
            route: AdministrationRoute::Oral,
            start_date: 1000,
            end_date: None,
        }
    }

    fn create_test_lab() -> LabResult {
        LabResult {
            test_code: "2345-7".to_string(),
            test_name: "Glucose".to_string(),
            value: 100.0,
            unit: "mg/dL".to_string(),
            reference_low: 70.0,
            reference_high: 100.0,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_health_link_patient_encoding() {
        let encoder = HealthLink::new();
        let patient = create_test_patient();

        let hv = encoder.encode_patient(&patient);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_health_link_vitals_encoding() {
        let encoder = HealthLink::new();
        let vitals = create_test_vitals();

        let hv = encoder.encode_vitals(&vitals);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_health_link_diagnosis_encoding() {
        let mut encoder = HealthLink::new();
        let diagnosis = create_test_diagnosis();

        let hv = encoder.encode_diagnosis(&diagnosis);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_health_link_medication_encoding() {
        let mut encoder = HealthLink::new();
        let medication = create_test_medication();

        let hv = encoder.encode_medication(&medication);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_lab_result_normal_check() {
        let lab = create_test_lab();
        assert!(lab.is_normal());
        assert!(!lab.is_critical());

        let critical_lab = LabResult {
            test_code: "2345-7".to_string(),
            test_name: "Glucose".to_string(),
            value: 500.0,
            unit: "mg/dL".to_string(),
            reference_low: 70.0,
            reference_high: 100.0,
            timestamp: 1000,
        };
        assert!(!critical_lab.is_normal());
        assert!(critical_lab.is_critical());
    }

    #[test]
    fn test_patient_database() {
        let mut db = PatientDatabase::new();

        let record = PatientRecord {
            patient: create_test_patient(),
            diagnoses: vec![create_test_diagnosis()],
            medications: vec![create_test_medication()],
            encounters: Vec::new(),
        };

        db.add_patient(record);
        assert_eq!(db.patient_count(), 1);
    }

    #[test]
    fn test_similar_patient_search() {
        let mut db = PatientDatabase::new();

        let patient1 = Patient {
            patient_id: "P001".to_string(),
            age: 65,
            sex: Sex::Male,
            blood_type: BloodType::APositive,
            weight_kg: 80.0,
            height_cm: 175.0,
        };

        let patient2 = Patient {
            patient_id: "P002".to_string(),
            age: 67,
            sex: Sex::Male,
            blood_type: BloodType::APositive,
            weight_kg: 82.0,
            height_cm: 177.0,
        };

        let diagnosis = create_test_diagnosis();

        db.add_patient(PatientRecord {
            patient: patient1,
            diagnoses: vec![diagnosis.clone()],
            medications: vec![],
            encounters: vec![],
        });

        db.add_patient(PatientRecord {
            patient: patient2,
            diagnoses: vec![diagnosis],
            medications: vec![],
            encounters: vec![],
        });

        let similar = db.find_similar("P001", 0.3, 10);
        assert!(!similar.is_empty());
        assert_eq!(similar[0].0, "P002");
    }

    #[test]
    fn test_clinical_decision_support() {
        let mut cds = ClinicalDecisionSupport::new();

        let symptoms = vec![Symptom {
            name: "Chest pain".to_string(),
            body_system: BodySystem::Cardiovascular,
            severity: Severity::Moderate,
            onset_date: 1000,
            duration_days: Some(1),
        }];

        let diagnosis = Diagnosis {
            icd_code: "I20.9".to_string(),
            description: "Angina pectoris".to_string(),
            diagnosed_date: 1000,
            is_primary: true,
            severity: Severity::Moderate,
        };

        cds.learn_case("C001", &symptoms, &[], &[diagnosis]);
        assert_eq!(cds.diagnosis_count(), 1);
    }

    #[test]
    fn test_drug_interaction_checker() {
        let mut checker = DrugInteractionChecker::new();

        checker.register_interaction("11289", "1191", InteractionSeverity::Major);
        assert_eq!(checker.interaction_count(), 1);

        let meds = vec![
            Medication {
                rxcui: "11289".to_string(),
                name: "Warfarin".to_string(),
                dose: 5.0,
                unit: "mg".to_string(),
                frequency: MedFrequency::Daily,
                route: AdministrationRoute::Oral,
                start_date: 1000,
                end_date: None,
            },
            Medication {
                rxcui: "1191".to_string(),
                name: "Aspirin".to_string(),
                dose: 81.0,
                unit: "mg".to_string(),
                frequency: MedFrequency::Daily,
                route: AdministrationRoute::Oral,
                start_date: 1000,
                end_date: None,
            },
        ];

        let interactions = checker.check_interactions(&meds);
        assert!(!interactions.is_empty());
        assert_eq!(interactions[0].severity, InteractionSeverity::Major);
    }

    #[test]
    fn test_risk_predictor() {
        let mut predictor = RiskPredictor::new();

        let encounter = MedicalEncounter {
            encounter_id: "E001".to_string(),
            patient_id: "P001".to_string(),
            encounter_date: 1000,
            encounter_type: EncounterType::Emergency,
            vitals: Some(Vitals {
                timestamp: 1000,
                heart_rate: 100,
                systolic_bp: 190,
                diastolic_bp: 100,
                temperature: 38.5,
                respiratory_rate: 22,
                oxygen_saturation: 92,
            }),
            symptoms: vec![],
            diagnoses: vec![
                create_test_diagnosis(),
                Diagnosis {
                    icd_code: "E11".to_string(),
                    description: "Type 2 diabetes".to_string(),
                    diagnosed_date: 1000,
                    is_primary: false,
                    severity: Severity::Moderate,
                },
                Diagnosis {
                    icd_code: "I50".to_string(),
                    description: "Heart failure".to_string(),
                    diagnosed_date: 1000,
                    is_primary: false,
                    severity: Severity::Moderate,
                },
            ],
            medications: vec![],
            procedures: vec![],
            lab_results: vec![],
        };

        predictor.train(&encounter, RiskOutcome::Readmission30Day);
        assert_eq!(predictor.training_count(), 1);
    }

    #[test]
    fn test_encounter_encoding() {
        let mut encoder = HealthLink::new();

        let encounter = MedicalEncounter {
            encounter_id: "E001".to_string(),
            patient_id: "P001".to_string(),
            encounter_date: 1000,
            encounter_type: EncounterType::Outpatient,
            vitals: Some(create_test_vitals()),
            symptoms: vec![Symptom {
                name: "Headache".to_string(),
                body_system: BodySystem::Neurological,
                severity: Severity::Mild,
                onset_date: 1000,
                duration_days: Some(2),
            }],
            diagnoses: vec![create_test_diagnosis()],
            medications: vec![create_test_medication()],
            procedures: vec![],
            lab_results: vec![create_test_lab()],
        };

        let hv = encoder.encode_encounter(&encounter);
        assert_eq!(hv.dimension(), DIMENSION);
    }
}
