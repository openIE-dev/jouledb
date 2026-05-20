//! JouleDB Auto Link
//!
//! HDC-powered Autonomous Vehicles and Robotics module.
//! Provides sensor fusion, object recognition, localization, and multi-agent coordination.

pub use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

/// Object classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ObjectClass {
    Vehicle,
    Pedestrian,
    Cyclist,
    Motorcycle,
    Truck,
    Bus,
    TrafficSign,
    TrafficLight,
    Obstacle,
    Lane,
    RoadMarking,
    Building,
    Vegetation,
    Animal,
    Unknown,
}

/// A detected object in the scene
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectedObject {
    pub object_id: String,
    pub class: ObjectClass,
    pub confidence: f64,
    pub bounding_box: BoundingBox3D,
    pub velocity: Option<Velocity3D>,
    pub attributes: Vec<ObjectAttribute>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BoundingBox3D {
    pub x: f64,      // Center X (meters from ego)
    pub y: f64,      // Center Y (meters from ego)
    pub z: f64,      // Center Z (height)
    pub length: f64, // Object length
    pub width: f64,  // Object width
    pub height: f64, // Object height
    pub yaw: f64,    // Heading angle (radians)
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Velocity3D {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ObjectAttribute {
    Moving,
    Stationary,
    Parked,
    Occluded,
    Truncated,
    Oncoming,
    SameDirection,
    CrossingPath,
    Emergency,
}

/// Sensor data from various sources
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SensorFrame {
    pub frame_id: u64,
    pub sensor_type: SensorType,
    pub timestamp: u64,
    pub detections: Vec<DetectedObject>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SensorType {
    Camera,
    Lidar,
    Radar,
    Ultrasonic,
    Imu,
    Gps,
    Fused,
}

/// A landmark for localization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Landmark {
    pub landmark_id: String,
    pub landmark_type: LandmarkType,
    pub position: Position3D,
    pub descriptor: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LandmarkType {
    SignPost,
    Building,
    Tree,
    Pole,
    Barrier,
    Curb,
    Crosswalk,
    StopLine,
    Custom,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Position3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Vehicle/robot state
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EgoState {
    pub position: Position3D,
    pub orientation: Orientation,
    pub velocity: Velocity3D,
    pub acceleration: Velocity3D,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Orientation {
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
}

/// Path/trajectory
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Trajectory {
    pub trajectory_id: String,
    pub waypoints: Vec<Waypoint>,
    pub total_distance: f64,
    pub estimated_time: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Waypoint {
    pub position: Position3D,
    pub velocity: f64,
    pub timestamp: u64,
}

/// Multi-agent coordination message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentState {
    pub agent_id: String,
    pub agent_type: AgentType,
    pub position: Position3D,
    pub velocity: Velocity3D,
    pub intent: Intent,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AgentType {
    AutonomousVehicle,
    Robot,
    Drone,
    Human,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Intent {
    GoStraight,
    TurnLeft,
    TurnRight,
    Stop,
    Yield,
    Park,
    EmergencyStop,
    Unknown,
}

// ============================================================================
// Auto Link Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// VSA Encoder for autonomous systems
    pub struct AutoLink {
        seed: 0xA070_D1E0,
        dimension: 10000,
        fields: ["class", "position", "velocity", "attributes", "sensor", "landmark", "intent", "agent"],
        scalars: ["distance", "angle", "speed", "confidence"],
        enums: {
            class_vectors: ObjectClass => [ObjectClass::Vehicle, ObjectClass::Pedestrian, ObjectClass::Cyclist, ObjectClass::Motorcycle, ObjectClass::Truck, ObjectClass::Bus, ObjectClass::TrafficSign, ObjectClass::TrafficLight, ObjectClass::Obstacle, ObjectClass::Lane, ObjectClass::RoadMarking, ObjectClass::Building, ObjectClass::Vegetation, ObjectClass::Animal, ObjectClass::Unknown],
            attribute_vectors: ObjectAttribute => [ObjectAttribute::Moving, ObjectAttribute::Stationary, ObjectAttribute::Parked, ObjectAttribute::Occluded, ObjectAttribute::Truncated, ObjectAttribute::Oncoming, ObjectAttribute::SameDirection, ObjectAttribute::CrossingPath, ObjectAttribute::Emergency],
            sensor_vectors: SensorType => [SensorType::Camera, SensorType::Lidar, SensorType::Radar, SensorType::Ultrasonic, SensorType::Imu, SensorType::Gps, SensorType::Fused],
            landmark_vectors: LandmarkType => [LandmarkType::SignPost, LandmarkType::Building, LandmarkType::Tree, LandmarkType::Pole, LandmarkType::Barrier, LandmarkType::Curb, LandmarkType::Crosswalk, LandmarkType::StopLine, LandmarkType::Custom],
            intent_vectors: Intent => [Intent::GoStraight, Intent::TurnLeft, Intent::TurnRight, Intent::Stop, Intent::Yield, Intent::Park, Intent::EmergencyStop, Intent::Unknown],
            agent_type_vectors: AgentType => [AgentType::AutonomousVehicle, AgentType::Robot, AgentType::Drone, AgentType::Human]
        },
    }
}

impl AutoLink {
    /// Encode a detected object
    pub fn encode_object(&self, obj: &DetectedObject) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Class
        acc.add(&self.field_vectors["class"].bind(&self.class_vectors[&obj.class]));

        // Position (distance and angle encoding)
        let distance = (obj.bounding_box.x.powi(2) + obj.bounding_box.y.powi(2)).sqrt();
        let dist_shift = (distance * 10.0) as usize % 157; // 0.1m resolution
        let dist_vec = self.scalar_bases["distance"].permute_words(dist_shift);
        acc.add(&self.field_vectors["position"].bind(&dist_vec));

        // Angle from ego
        let angle = obj.bounding_box.y.atan2(obj.bounding_box.x);
        let angle_shift = ((angle + std::f64::consts::PI) * 25.0) as usize % 157;
        let angle_vec = self.scalar_bases["angle"].permute_words(angle_shift);
        acc.add(&angle_vec);

        // Velocity magnitude
        if let Some(vel) = &obj.velocity {
            let speed = (vel.vx.powi(2) + vel.vy.powi(2)).sqrt();
            let speed_shift = (speed * 10.0) as usize % 157;
            let speed_vec = self.scalar_bases["speed"].permute_words(speed_shift);
            acc.add(&self.field_vectors["velocity"].bind(&speed_vec));
        }

        // Attributes
        for attr in &obj.attributes {
            acc.add(&self.attribute_vectors[attr]);
        }

        // Confidence
        let conf_shift = (obj.confidence * 100.0) as usize % 157;
        let conf_vec = self.scalar_bases["confidence"].permute_words(conf_shift);
        acc.add(&conf_vec);

        acc.threshold()
    }

    /// Encode an entire scene (all objects fused)
    pub fn encode_scene(&self, objects: &[DetectedObject]) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        for obj in objects {
            acc.add(&self.encode_object(obj));
        }

        acc.threshold()
    }

    /// Encode a landmark for localization
    pub fn encode_landmark(&self, landmark: &Landmark) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Type
        acc.add(
            &self.field_vectors["landmark"].bind(&self.landmark_vectors[&landmark.landmark_type]),
        );

        // Position
        let dist = (landmark.position.x.powi(2) + landmark.position.y.powi(2)).sqrt();
        let dist_shift = (dist * 10.0) as usize % 157;
        let dist_vec = self.scalar_bases["distance"].permute_words(dist_shift);
        acc.add(&self.field_vectors["position"].bind(&dist_vec));

        // Descriptor (if available)
        for (i, &val) in landmark.descriptor.iter().take(10).enumerate() {
            let val_shift = (val * 100.0).abs() as usize % 157;
            let val_vec = self.scalar_bases["distance"].permute_words(val_shift + i);
            acc.add(&val_vec);
        }

        acc.threshold()
    }

    /// Encode an agent state for multi-agent coordination
    pub fn encode_agent(&self, agent: &AgentState) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Agent type
        acc.add(&self.field_vectors["agent"].bind(&self.agent_type_vectors[&agent.agent_type]));

        // Position
        let dist = (agent.position.x.powi(2) + agent.position.y.powi(2)).sqrt();
        let dist_shift = (dist * 10.0) as usize % 157;
        let dist_vec = self.scalar_bases["distance"].permute_words(dist_shift);
        acc.add(&self.field_vectors["position"].bind(&dist_vec));

        // Intent
        acc.add(&self.field_vectors["intent"].bind(&self.intent_vectors[&agent.intent]));

        // Speed
        let speed = (agent.velocity.vx.powi(2) + agent.velocity.vy.powi(2)).sqrt();
        let speed_shift = (speed * 10.0) as usize % 157;
        let speed_vec = self.scalar_bases["speed"].permute_words(speed_shift);
        acc.add(&self.field_vectors["velocity"].bind(&speed_vec));

        acc.threshold()
    }
}

// ============================================================================
// Object Recognition
// ============================================================================

/// Holographic object library for recognition
pub struct ObjectLibrary {
    /// Known object patterns by class
    class_bundles: HashMap<ObjectClass, BundleAccumulator>,
    /// Individual known objects
    known_objects: HashMap<String, BinaryHV>,
    /// Encoder
    encoder: AutoLink,
}

impl ObjectLibrary {
    pub fn new() -> Self {
        Self {
            class_bundles: HashMap::new(),
            known_objects: HashMap::new(),
            encoder: AutoLink::new(),
        }
    }

    /// Add an object to the library
    pub fn add_object(&mut self, obj: &DetectedObject) {
        let hv = self.encoder.encode_object(obj);

        // Add to class bundle
        let bundle = self
            .class_bundles
            .entry(obj.class)
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        // Store individual
        self.known_objects.insert(obj.object_id.clone(), hv);
    }

    /// Classify an unknown object by similarity to known classes
    pub fn classify(&self, obj: &DetectedObject) -> Vec<(ObjectClass, f32)> {
        let obj_hv = self.encoder.encode_object(obj);

        let mut scores: Vec<(ObjectClass, f32)> = self
            .class_bundles
            .iter()
            .map(|(class, bundle)| (*class, obj_hv.similarity(&bundle.threshold())))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Find similar known objects
    pub fn find_similar(&self, obj: &DetectedObject, threshold: f32) -> Vec<(String, f32)> {
        let obj_hv = self.encoder.encode_object(obj);

        let mut matches: Vec<(String, f32)> = self
            .known_objects
            .iter()
            .map(|(id, hv)| (id.clone(), obj_hv.similarity(hv)))
            .filter(|(_, sim)| *sim > threshold)
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }
}

impl Default for ObjectLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Localization
// ============================================================================

/// Holographic map for localization
pub struct HolographicMap {
    /// Landmark bundles by type
    landmark_bundles: HashMap<LandmarkType, BundleAccumulator>,
    /// All landmarks
    landmarks: HashMap<String, (Landmark, BinaryHV)>,
    /// Encoder
    encoder: AutoLink,
}

impl HolographicMap {
    pub fn new() -> Self {
        Self {
            landmark_bundles: HashMap::new(),
            landmarks: HashMap::new(),
            encoder: AutoLink::new(),
        }
    }

    /// Add a landmark to the map
    pub fn add_landmark(&mut self, landmark: Landmark) {
        let hv = self.encoder.encode_landmark(&landmark);

        let bundle = self
            .landmark_bundles
            .entry(landmark.landmark_type)
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        self.landmarks
            .insert(landmark.landmark_id.clone(), (landmark, hv));
    }

    /// Match observed landmarks to map
    pub fn match_landmarks(&self, observed: &[Landmark]) -> Vec<LandmarkMatch> {
        let mut matches = Vec::new();

        for obs in observed {
            let obs_hv = self.encoder.encode_landmark(obs);

            // Find best match
            let mut best: Option<(String, f32)> = None;

            for (id, (_, map_hv)) in &self.landmarks {
                let sim = obs_hv.similarity(map_hv);
                if sim > 0.6 {
                    if best.is_none() || sim > best.as_ref().unwrap().1 {
                        best = Some((id.clone(), sim));
                    }
                }
            }

            if let Some((landmark_id, similarity)) = best {
                matches.push(LandmarkMatch {
                    observed_id: obs.landmark_id.clone(),
                    map_landmark_id: landmark_id,
                    similarity,
                });
            }
        }

        matches
    }

    /// Localize based on observed landmarks
    pub fn localize(&self, observed: &[Landmark]) -> Option<LocalizationResult> {
        let matches = self.match_landmarks(observed);

        if matches.len() < 3 {
            return None; // Need at least 3 matches
        }

        // Simple averaging of matched landmark positions
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_weight = 0.0;

        for m in &matches {
            if let Some((landmark, _)) = self.landmarks.get(&m.map_landmark_id) {
                let weight = m.similarity as f64;
                sum_x += landmark.position.x * weight;
                sum_y += landmark.position.y * weight;
                sum_weight += weight;
            }
        }

        if sum_weight > 0.0 {
            Some(LocalizationResult {
                position: Position3D {
                    x: sum_x / sum_weight,
                    y: sum_y / sum_weight,
                    z: 0.0,
                },
                confidence: (sum_weight / matches.len() as f64) as f32,
                matched_landmarks: matches.len(),
            })
        } else {
            None
        }
    }

    /// Landmark count
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }
}

impl Default for HolographicMap {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct LandmarkMatch {
    pub observed_id: String,
    pub map_landmark_id: String,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct LocalizationResult {
    pub position: Position3D,
    pub confidence: f32,
    pub matched_landmarks: usize,
}

// ============================================================================
// Multi-Agent Coordination
// ============================================================================

/// Multi-agent coordination using holographic state sharing
pub struct AgentCoordinator {
    /// Agent states
    agents: HashMap<String, (AgentState, BinaryHV)>,
    /// Intent-position bundles for collision detection
    intent_bundles: HashMap<Intent, BundleAccumulator>,
    /// Encoder
    encoder: AutoLink,
}

impl AgentCoordinator {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            intent_bundles: HashMap::new(),
            encoder: AutoLink::new(),
        }
    }

    /// Update an agent's state
    pub fn update_agent(&mut self, state: AgentState) {
        let hv = self.encoder.encode_agent(&state);

        // Add to intent bundle
        let bundle = self
            .intent_bundles
            .entry(state.intent)
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        self.agents.insert(state.agent_id.clone(), (state, hv));
    }

    /// Check for potential conflicts with other agents
    pub fn check_conflicts(&self, ego: &AgentState) -> Vec<ConflictWarning> {
        let ego_hv = self.encoder.encode_agent(ego);
        let mut warnings = Vec::new();

        for (id, (other, other_hv)) in &self.agents {
            if id == &ego.agent_id {
                continue;
            }

            // Check similarity (similar position/intent = potential conflict)
            let similarity = ego_hv.similarity(other_hv);

            if similarity > 0.7 {
                // Calculate distance
                let dist = ((ego.position.x - other.position.x).powi(2)
                    + (ego.position.y - other.position.y).powi(2))
                .sqrt();

                if dist < 50.0 {
                    warnings.push(ConflictWarning {
                        other_agent_id: id.clone(),
                        similarity,
                        distance: dist,
                        other_intent: other.intent,
                        severity: if dist < 10.0 {
                            ConflictSeverity::Critical
                        } else if dist < 25.0 {
                            ConflictSeverity::Warning
                        } else {
                            ConflictSeverity::Advisory
                        },
                    });
                }
            }
        }

        warnings
    }

    /// Get all agents with a specific intent
    pub fn agents_with_intent(&self, intent: Intent) -> Vec<&AgentState> {
        self.agents
            .values()
            .filter(|(s, _)| s.intent == intent)
            .map(|(s, _)| s)
            .collect()
    }

    /// Agent count
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for AgentCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ConflictWarning {
    pub other_agent_id: String,
    pub similarity: f32,
    pub distance: f64,
    pub other_intent: Intent,
    pub severity: ConflictSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictSeverity {
    Advisory,
    Warning,
    Critical,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_object(class: ObjectClass, x: f64, y: f64) -> DetectedObject {
        DetectedObject {
            object_id: format!("obj_{}_{}", x as i32, y as i32),
            class,
            confidence: 0.95,
            bounding_box: BoundingBox3D {
                x,
                y,
                z: 0.0,
                length: 4.0,
                width: 2.0,
                height: 1.5,
                yaw: 0.0,
            },
            velocity: Some(Velocity3D {
                vx: 10.0,
                vy: 0.0,
                vz: 0.0,
            }),
            attributes: vec![ObjectAttribute::Moving],
            timestamp: 1000,
        }
    }

    #[test]
    fn test_object_encoding() {
        let link = AutoLink::new();

        let obj = make_object(ObjectClass::Vehicle, 10.0, 5.0);
        let hv = link.encode_object(&obj);

        // Same object should encode consistently
        let hv2 = link.encode_object(&obj);
        assert!((hv.similarity(&hv2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_object_classification() {
        let mut library = ObjectLibrary::new();

        // Add known vehicles
        for i in 0..20 {
            library.add_object(&make_object(ObjectClass::Vehicle, i as f64 * 10.0, 0.0));
        }

        // Add known pedestrians
        for i in 0..20 {
            let mut ped = make_object(ObjectClass::Pedestrian, i as f64 * 5.0, 10.0);
            ped.bounding_box.length = 0.5;
            ped.bounding_box.width = 0.5;
            ped.bounding_box.height = 1.8;
            library.add_object(&ped);
        }

        // Classify new vehicle
        let new_vehicle = make_object(ObjectClass::Unknown, 50.0, 0.0);
        let scores = library.classify(&new_vehicle);

        println!("Classification scores: {:?}", scores);
        assert!(!scores.is_empty());
    }

    #[test]
    fn test_landmark_matching() {
        let mut map = HolographicMap::new();

        // Add landmarks to map
        map.add_landmark(Landmark {
            landmark_id: "sign_001".to_string(),
            landmark_type: LandmarkType::SignPost,
            position: Position3D {
                x: 100.0,
                y: 50.0,
                z: 3.0,
            },
            descriptor: vec![1.0, 0.5, 0.2],
        });

        map.add_landmark(Landmark {
            landmark_id: "pole_001".to_string(),
            landmark_type: LandmarkType::Pole,
            position: Position3D {
                x: 120.0,
                y: 45.0,
                z: 5.0,
            },
            descriptor: vec![0.8, 0.3, 0.1],
        });

        // Observe similar landmarks
        let observed = vec![Landmark {
            landmark_id: "obs_1".to_string(),
            landmark_type: LandmarkType::SignPost,
            position: Position3D {
                x: 101.0,
                y: 51.0,
                z: 3.0,
            },
            descriptor: vec![1.0, 0.5, 0.2],
        }];

        let matches = map.match_landmarks(&observed);
        println!("Landmark matches: {:?}", matches);
    }

    #[test]
    fn test_agent_coordination() {
        let mut coordinator = AgentCoordinator::new();

        // Agent 1 going straight
        coordinator.update_agent(AgentState {
            agent_id: "av_001".to_string(),
            agent_type: AgentType::AutonomousVehicle,
            position: Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: Velocity3D {
                vx: 10.0,
                vy: 0.0,
                vz: 0.0,
            },
            intent: Intent::GoStraight,
            timestamp: 1000,
        });

        // Agent 2 approaching from side
        coordinator.update_agent(AgentState {
            agent_id: "av_002".to_string(),
            agent_type: AgentType::AutonomousVehicle,
            position: Position3D {
                x: 20.0,
                y: 15.0,
                z: 0.0,
            },
            velocity: Velocity3D {
                vx: 0.0,
                vy: -10.0,
                vz: 0.0,
            },
            intent: Intent::TurnLeft,
            timestamp: 1000,
        });

        // Check conflicts for agent 1
        let ego = AgentState {
            agent_id: "av_001".to_string(),
            agent_type: AgentType::AutonomousVehicle,
            position: Position3D {
                x: 15.0,
                y: 0.0,
                z: 0.0,
            },
            velocity: Velocity3D {
                vx: 10.0,
                vy: 0.0,
                vz: 0.0,
            },
            intent: Intent::GoStraight,
            timestamp: 1001,
        };

        let warnings = coordinator.check_conflicts(&ego);
        println!("Conflict warnings: {:?}", warnings);
    }
}
