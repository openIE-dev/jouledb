//! HDC-powered Education and EdTech Learning module
//!
//! Provides holographic encoding for:
//! - Learning path optimization
//! - Student similarity and cohort analysis
//! - Content recommendation
//! - Plagiarism detection

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DIMENSION: usize = 10000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Subject {
    Math,
    Science,
    English,
    History,
    Art,
    Music,
    CS,
    Languages,
    Business,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Student {
    pub id: String,
    pub grade_level: u8,
    pub subjects: Vec<Subject>,
    pub gpa: f32,
    pub learning_style: String,
    pub completed_courses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Course {
    pub id: String,
    pub title: String,
    pub subject: Subject,
    pub difficulty: SkillLevel,
    pub prerequisites: Vec<String>,
    pub learning_outcomes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    pub id: String,
    pub student_id: String,
    pub course_id: String,
    pub content: String,
    pub score: f32,
    pub submission_time: u64,
}

joule_db_hdc::define_domain_module! {
    /// HDC encoder for education domain data
    pub struct EduLink {
        seed: 0xED00_0001,
        dimension: 10000,
        fields: ["student", "course", "subject", "level", "content", "score", "outcome"],
        scalars: ["grade", "gpa", "score", "time"],
        enums: {
            subject_vectors: Subject => [Subject::Math, Subject::Science, Subject::English, Subject::History, Subject::Art, Subject::Music, Subject::CS, Subject::Languages, Subject::Business],
            level_vectors: SkillLevel => [SkillLevel::Beginner, SkillLevel::Intermediate, SkillLevel::Advanced, SkillLevel::Expert]
        },
        dynamic: {
            style_vectors: "style"
        },
    }
}

impl EduLink {
    pub fn encode_student(&mut self, student: &Student) -> BinaryHV {
        let grade_hv = self.encode_scalar("grade", student.grade_level as u32, 12);
        let gpa_hv = self.encode_scalar("gpa", (student.gpa * 100.0) as u32, 400);
        let style_vec = self.style_vectors(&student.learning_style);
        let style_hv = self.field_vectors["student"].bind(&style_vec);
        let mut components = vec![grade_hv, gpa_hv, style_hv];
        for subject in &student.subjects {
            components.push(self.field_vectors["subject"].bind(&self.subject_vectors[subject]));
        }
        self.bundle(&components)
    }

    pub fn encode_course(&self, course: &Course) -> BinaryHV {
        let subject_hv = self.field_vectors["subject"].bind(&self.subject_vectors[&course.subject]);
        let level_hv = self.field_vectors["level"].bind(&self.level_vectors[&course.difficulty]);
        let title_hv = BinaryHV::from_hash(course.title.as_bytes(), DIMENSION);
        let mut components = vec![subject_hv, level_hv, title_hv];
        for outcome in &course.learning_outcomes {
            components.push(
                self.field_vectors["outcome"]
                    .bind(&BinaryHV::from_hash(outcome.as_bytes(), DIMENSION)),
            );
        }
        self.bundle(&components)
    }

    pub fn encode_assignment(&self, assignment: &Assignment) -> BinaryHV {
        let content_hv = BinaryHV::from_hash(assignment.content.as_bytes(), DIMENSION);
        let score_hv = self.encode_scalar("score", (assignment.score * 100.0) as u32, 100);
        self.bundle(&[content_hv, score_hv])
    }
}

pub struct LearningPathOptimizer {
    encoder: EduLink,
    course_vectors: HashMap<String, BinaryHV>,
    courses: HashMap<String, Course>,
}

impl LearningPathOptimizer {
    pub fn new() -> Self {
        Self {
            encoder: EduLink::new(),
            course_vectors: HashMap::new(),
            courses: HashMap::new(),
        }
    }

    pub fn add_course(&mut self, course: Course) {
        let hv = self.encoder.encode_course(&course);
        self.course_vectors.insert(course.id.clone(), hv);
        self.courses.insert(course.id.clone(), course);
    }

    pub fn recommend(&mut self, student: &Student, limit: usize) -> Vec<(String, f32)> {
        let student_hv = self.encoder.encode_student(student);
        let completed: std::collections::HashSet<_> = student.completed_courses.iter().collect();
        let mut results: Vec<_> = self
            .course_vectors
            .iter()
            .filter(|(id, _)| !completed.contains(*id))
            .map(|(id, hv)| (id.clone(), student_hv.similarity(hv)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn course_count(&self) -> usize {
        self.courses.len()
    }
}

impl Default for LearningPathOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PlagiarismDetector {
    encoder: EduLink,
    document_vectors: HashMap<String, BinaryHV>,
    threshold: f32,
}

impl PlagiarismDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            encoder: EduLink::new(),
            document_vectors: HashMap::new(),
            threshold,
        }
    }

    pub fn index_assignment(&mut self, assignment: &Assignment) {
        let hv = self.encoder.encode_assignment(assignment);
        self.document_vectors.insert(assignment.id.clone(), hv);
    }

    pub fn check(&self, assignment: &Assignment) -> Vec<(String, f32)> {
        let hv = self.encoder.encode_assignment(assignment);
        self.document_vectors
            .iter()
            .filter(|(id, _)| *id != &assignment.id)
            .map(|(id, doc_hv)| (id.clone(), hv.similarity(doc_hv)))
            .filter(|(_, sim)| *sim >= self.threshold)
            .collect()
    }
}

impl Default for PlagiarismDetector {
    fn default() -> Self {
        Self::new(0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_student_encoding() {
        let mut encoder = EduLink::new();
        let student = Student {
            id: "S1".to_string(),
            grade_level: 10,
            subjects: vec![Subject::Math, Subject::CS],
            gpa: 3.5,
            learning_style: "visual".to_string(),
            completed_courses: vec![],
        };
        assert_eq!(encoder.encode_student(&student).dimension(), DIMENSION);
    }

    #[test]
    fn test_course_recommendation() {
        let mut optimizer = LearningPathOptimizer::new();
        optimizer.add_course(Course {
            id: "C1".to_string(),
            title: "Intro to CS".to_string(),
            subject: Subject::CS,
            difficulty: SkillLevel::Beginner,
            prerequisites: vec![],
            learning_outcomes: vec!["programming".to_string()],
        });
        let mut encoder = EduLink::new();
        let student = Student {
            id: "S1".to_string(),
            grade_level: 10,
            subjects: vec![Subject::CS],
            gpa: 3.5,
            learning_style: "visual".to_string(),
            completed_courses: vec![],
        };
        let recs = optimizer.recommend(&student, 5);
        assert!(!recs.is_empty());
    }

    #[test]
    fn test_plagiarism_detection() {
        let mut detector = PlagiarismDetector::new(0.8);
        let a1 = Assignment {
            id: "A1".to_string(),
            student_id: "S1".to_string(),
            course_id: "C1".to_string(),
            content: "This is my essay about science.".to_string(),
            score: 0.9,
            submission_time: 0,
        };
        detector.index_assignment(&a1);
        assert_eq!(detector.document_vectors.len(), 1);
    }
}
