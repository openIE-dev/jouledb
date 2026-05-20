//! Disk scheduling algorithms — FCFS, SSTF, SCAN (elevator), C-SCAN,
//! LOOK, C-LOOK. Seek distance calculation, throughput comparison,
//! request queue management, arm position tracking.

use std::collections::VecDeque;

// ── Algorithm ───────────────────────────────────────────────────────────────

/// Disk scheduling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskAlgorithm {
    Fcfs,
    Sstf,
    Scan,
    CScan,
    Look,
    CLook,
}

impl std::fmt::Display for DiskAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiskAlgorithm::Fcfs => write!(f, "FCFS"),
            DiskAlgorithm::Sstf => write!(f, "SSTF"),
            DiskAlgorithm::Scan => write!(f, "SCAN"),
            DiskAlgorithm::CScan => write!(f, "C-SCAN"),
            DiskAlgorithm::Look => write!(f, "LOOK"),
            DiskAlgorithm::CLook => write!(f, "C-LOOK"),
        }
    }
}

// ── Direction ───────────────────────────────────────────────────────────────

/// Direction the disk arm is moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
}

// ── Request ─────────────────────────────────────────────────────────────────

/// A disk I/O request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskRequest {
    pub cylinder: u32,
    pub arrival_order: u32,
}

// ── Schedule Result ─────────────────────────────────────────────────────────

/// Result of scheduling a set of requests.
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    pub algorithm: DiskAlgorithm,
    /// Order in which cylinders are visited.
    pub visit_order: Vec<u32>,
    /// Total seek distance (sum of absolute differences).
    pub total_seek: u64,
    /// Individual seek distances between consecutive visits.
    pub seek_sequence: Vec<u64>,
    /// Average seek distance per request.
    pub avg_seek: f64,
    /// Maximum single seek.
    pub max_seek: u64,
}

// ── DiskScheduler ───────────────────────────────────────────────────────────

/// Disk arm scheduler simulation.
#[derive(Debug)]
pub struct DiskScheduler {
    /// Total number of cylinders on the disk (0..max_cylinder-1).
    max_cylinder: u32,
    /// Current arm position.
    arm_position: u32,
    /// Current direction (for SCAN/LOOK variants).
    direction: Direction,
    /// Pending request queue.
    queue: VecDeque<DiskRequest>,
    /// Monotonic counter for arrival ordering.
    arrival_counter: u32,
    /// History of arm positions visited.
    history: Vec<u32>,
    /// Total seek accumulated.
    total_seek: u64,
    /// Number of requests served.
    served_count: u64,
}

impl DiskScheduler {
    /// Create a new scheduler for a disk with `max_cylinder` cylinders.
    pub fn new(max_cylinder: u32, initial_position: u32) -> Self {
        Self {
            max_cylinder,
            arm_position: initial_position,
            direction: Direction::Up,
            queue: VecDeque::new(),
            arrival_counter: 0,
            history: vec![initial_position],
            total_seek: 0,
            served_count: 0,
        }
    }

    /// Set arm direction.
    pub fn set_direction(&mut self, dir: Direction) {
        self.direction = dir;
    }

    /// Current arm position.
    pub fn arm_position(&self) -> u32 {
        self.arm_position
    }

    /// Current direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Add a request to the queue.
    pub fn add_request(&mut self, cylinder: u32) {
        let order = self.arrival_counter;
        self.arrival_counter += 1;
        self.queue.push_back(DiskRequest {
            cylinder,
            arrival_order: order,
        });
    }

    /// Add multiple requests.
    pub fn add_requests(&mut self, cylinders: &[u32]) {
        for &c in cylinders {
            self.add_request(c);
        }
    }

    /// Number of pending requests.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Process all pending requests using the given algorithm.
    /// Returns the schedule result.
    pub fn schedule(&mut self, algorithm: DiskAlgorithm) -> ScheduleResult {
        let requests: Vec<DiskRequest> = self.queue.drain(..).collect();
        if requests.is_empty() {
            return ScheduleResult {
                algorithm,
                visit_order: vec![],
                total_seek: 0,
                seek_sequence: vec![],
                avg_seek: 0.0,
                max_seek: 0,
            };
        }

        let visit_order = match algorithm {
            DiskAlgorithm::Fcfs => self.schedule_fcfs(&requests),
            DiskAlgorithm::Sstf => self.schedule_sstf(&requests),
            DiskAlgorithm::Scan => self.schedule_scan(&requests),
            DiskAlgorithm::CScan => self.schedule_cscan(&requests),
            DiskAlgorithm::Look => self.schedule_look(&requests),
            DiskAlgorithm::CLook => self.schedule_clook(&requests),
        };

        let (total_seek, seek_sequence, max_seek) =
            self.compute_seeks(&visit_order);

        let avg_seek = if !visit_order.is_empty() {
            total_seek as f64 / visit_order.len() as f64
        } else {
            0.0
        };

        // Update arm position to last visited
        if let Some(&last) = visit_order.last() {
            self.arm_position = last;
            self.history.push(last);
        }
        self.total_seek += total_seek;
        self.served_count += visit_order.len() as u64;

        ScheduleResult {
            algorithm,
            visit_order,
            total_seek,
            seek_sequence,
            avg_seek,
            max_seek,
        }
    }

    fn compute_seeks(&self, visit_order: &[u32]) -> (u64, Vec<u64>, u64) {
        let mut total = 0u64;
        let mut seeks = Vec::new();
        let mut max = 0u64;
        let mut pos = self.arm_position;

        for &cyl in visit_order {
            let seek = (pos as i64 - cyl as i64).unsigned_abs();
            total += seek;
            seeks.push(seek);
            if seek > max {
                max = seek;
            }
            pos = cyl;
        }
        (total, seeks, max)
    }

    // ── FCFS ──

    fn schedule_fcfs(&self, requests: &[DiskRequest]) -> Vec<u32> {
        let mut ordered: Vec<&DiskRequest> = requests.iter().collect();
        ordered.sort_by_key(|r| r.arrival_order);
        ordered.iter().map(|r| r.cylinder).collect()
    }

    // ── SSTF ──

    fn schedule_sstf(&self, requests: &[DiskRequest]) -> Vec<u32> {
        let mut remaining: Vec<u32> = requests.iter().map(|r| r.cylinder).collect();
        let mut result = Vec::with_capacity(remaining.len());
        let mut pos = self.arm_position;

        while !remaining.is_empty() {
            let (idx, _) = remaining
                .iter()
                .enumerate()
                .min_by_key(|(_, cyl)| (pos as i64 - (**cyl) as i64).unsigned_abs())
                .unwrap();
            let cyl = remaining.remove(idx);
            result.push(cyl);
            pos = cyl;
        }
        result
    }

    // ── SCAN (Elevator) ──

    fn schedule_scan(&self, requests: &[DiskRequest]) -> Vec<u32> {
        let mut cylinders: Vec<u32> = requests.iter().map(|r| r.cylinder).collect();
        cylinders.sort();
        cylinders.dedup();

        let pos = self.arm_position;
        let mut result = Vec::new();

        match self.direction {
            Direction::Up => {
                // Go up to max_cylinder-1, then down
                let up: Vec<u32> = cylinders.iter().copied().filter(|c| *c >= pos).collect();
                let down: Vec<u32> = cylinders.iter().copied().filter(|c| *c < pos).rev().collect();
                result.extend(up);
                if !down.is_empty() {
                    // SCAN goes to end of disk before reversing
                    if result.last().copied() != Some(self.max_cylinder - 1) {
                        result.push(self.max_cylinder - 1);
                    }
                    result.extend(down);
                }
            }
            Direction::Down => {
                let down: Vec<u32> = cylinders.iter().copied().filter(|c| *c <= pos).rev().collect();
                let up: Vec<u32> = cylinders.iter().copied().filter(|c| *c > pos).collect();
                result.extend(down);
                if !up.is_empty() {
                    if result.last().copied() != Some(0) {
                        result.push(0);
                    }
                    result.extend(up);
                }
            }
        }
        result
    }

    // ── C-SCAN ──

    fn schedule_cscan(&self, requests: &[DiskRequest]) -> Vec<u32> {
        let mut cylinders: Vec<u32> = requests.iter().map(|r| r.cylinder).collect();
        cylinders.sort();
        cylinders.dedup();

        let pos = self.arm_position;
        let mut result = Vec::new();

        // Always go up, then jump to beginning and continue up
        let up: Vec<u32> = cylinders.iter().copied().filter(|c| *c >= pos).collect();
        let wrap: Vec<u32> = cylinders.iter().copied().filter(|c| *c < pos).collect();

        result.extend(up);
        if !wrap.is_empty() {
            // Go to end, jump to 0, then serve remaining
            let last = result.last().copied().unwrap_or(pos);
            if last != self.max_cylinder - 1 {
                result.push(self.max_cylinder - 1);
            }
            result.push(0);
            result.extend(wrap);
        }
        result
    }

    // ── LOOK ──

    fn schedule_look(&self, requests: &[DiskRequest]) -> Vec<u32> {
        let mut cylinders: Vec<u32> = requests.iter().map(|r| r.cylinder).collect();
        cylinders.sort();
        cylinders.dedup();

        let pos = self.arm_position;
        let mut result = Vec::new();

        match self.direction {
            Direction::Up => {
                let up: Vec<u32> = cylinders.iter().copied().filter(|c| *c >= pos).collect();
                let down: Vec<u32> = cylinders.iter().copied().filter(|c| *c < pos).rev().collect();
                result.extend(up);
                result.extend(down);
            }
            Direction::Down => {
                let down: Vec<u32> = cylinders.iter().copied().filter(|c| *c <= pos).rev().collect();
                let up: Vec<u32> = cylinders.iter().copied().filter(|c| *c > pos).collect();
                result.extend(down);
                result.extend(up);
            }
        }
        result
    }

    // ── C-LOOK ──

    fn schedule_clook(&self, requests: &[DiskRequest]) -> Vec<u32> {
        let mut cylinders: Vec<u32> = requests.iter().map(|r| r.cylinder).collect();
        cylinders.sort();
        cylinders.dedup();

        let pos = self.arm_position;

        // Always go up, then jump to lowest and continue up
        let up: Vec<u32> = cylinders.iter().copied().filter(|c| *c >= pos).collect();
        let wrap: Vec<u32> = cylinders.iter().copied().filter(|c| *c < pos).collect();

        let mut result = Vec::new();
        result.extend(up);
        result.extend(wrap);
        result
    }

    /// Visit history.
    pub fn visit_history(&self) -> &[u32] {
        &self.history
    }

    /// Total seek distance accumulated across all schedules.
    pub fn cumulative_seek(&self) -> u64 {
        self.total_seek
    }

    /// Total requests served.
    pub fn total_served(&self) -> u64 {
        self.served_count
    }

    /// Reset the scheduler state (keeps max_cylinder).
    pub fn reset(&mut self, initial_position: u32) {
        self.arm_position = initial_position;
        self.direction = Direction::Up;
        self.queue.clear();
        self.arrival_counter = 0;
        self.history = vec![initial_position];
        self.total_seek = 0;
        self.served_count = 0;
    }
}

// ── Comparison ──────────────────────────────────────────────────────────────

/// Compare all algorithms on the same set of requests.
pub fn compare_algorithms(
    max_cylinder: u32,
    initial_position: u32,
    requests: &[u32],
    direction: Direction,
) -> Vec<ScheduleResult> {
    let algorithms = [
        DiskAlgorithm::Fcfs,
        DiskAlgorithm::Sstf,
        DiskAlgorithm::Scan,
        DiskAlgorithm::CScan,
        DiskAlgorithm::Look,
        DiskAlgorithm::CLook,
    ];

    algorithms
        .iter()
        .map(|alg| {
            let mut sched = DiskScheduler::new(max_cylinder, initial_position);
            sched.set_direction(direction);
            sched.add_requests(requests);
            sched.schedule(*alg)
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_requests() -> Vec<u32> {
        vec![98, 183, 37, 122, 14, 124, 65, 67]
    }

    #[test]
    fn test_fcfs_order() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.add_requests(&sample_requests());
        let result = sched.schedule(DiskAlgorithm::Fcfs);
        assert_eq!(result.visit_order, vec![98, 183, 37, 122, 14, 124, 65, 67]);
    }

    #[test]
    fn test_fcfs_seek_distance() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.add_requests(&sample_requests());
        let result = sched.schedule(DiskAlgorithm::Fcfs);
        assert!(result.total_seek > 0);
        assert_eq!(result.seek_sequence.len(), 8);
    }

    #[test]
    fn test_sstf_greedy() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.add_requests(&sample_requests());
        let result = sched.schedule(DiskAlgorithm::Sstf);
        // First move should be to the nearest cylinder
        assert_eq!(result.visit_order[0], 65); // 65 is closest to 53
    }

    #[test]
    fn test_sstf_total_seek() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.add_requests(&sample_requests());
        let result = sched.schedule(DiskAlgorithm::Sstf);
        // SSTF should be better than FCFS (usually)
        let mut sched2 = DiskScheduler::new(200, 53);
        sched2.add_requests(&sample_requests());
        let fcfs_result = sched2.schedule(DiskAlgorithm::Fcfs);
        assert!(result.total_seek <= fcfs_result.total_seek);
    }

    #[test]
    fn test_scan_visits_end() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.set_direction(Direction::Up);
        sched.add_requests(&[37, 98, 183]);
        let result = sched.schedule(DiskAlgorithm::Scan);
        // SCAN going up should visit cylinders >= 53 first, then reverse
        // Should visit 98, 183 going up, then 199 (end), then 37 coming down
        assert!(result.visit_order.contains(&98));
        assert!(result.visit_order.contains(&183));
        assert!(result.visit_order.contains(&37));
    }

    #[test]
    fn test_look_no_end() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.set_direction(Direction::Up);
        sched.add_requests(&[37, 98, 183]);
        let result = sched.schedule(DiskAlgorithm::Look);
        // LOOK should NOT go to cylinder 199 — only to highest request
        assert!(!result.visit_order.contains(&199));
    }

    #[test]
    fn test_cscan_wraps() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.add_requests(&[37, 98, 183, 14]);
        let result = sched.schedule(DiskAlgorithm::CScan);
        // C-SCAN always goes in one direction, then wraps
        // After serving upper requests, it should wrap to 0 and continue
        let has_wrap = result.visit_order.contains(&0);
        // If there are requests below current position, it should wrap
        assert!(has_wrap);
    }

    #[test]
    fn test_clook_no_zero() {
        let mut sched = DiskScheduler::new(200, 53);
        sched.add_requests(&[37, 98, 183, 14]);
        let result = sched.schedule(DiskAlgorithm::CLook);
        // C-LOOK should not go to 0, just jump to lowest request
        assert!(!result.visit_order.contains(&0));
    }

    #[test]
    fn test_empty_queue() {
        let mut sched = DiskScheduler::new(200, 53);
        let result = sched.schedule(DiskAlgorithm::Fcfs);
        assert_eq!(result.total_seek, 0);
        assert!(result.visit_order.is_empty());
    }

    #[test]
    fn test_single_request() {
        let mut sched = DiskScheduler::new(200, 50);
        sched.add_request(100);
        let result = sched.schedule(DiskAlgorithm::Fcfs);
        assert_eq!(result.visit_order, vec![100]);
        assert_eq!(result.total_seek, 50);
    }

    #[test]
    fn test_arm_position_updates() {
        let mut sched = DiskScheduler::new(200, 50);
        sched.add_request(100);
        sched.schedule(DiskAlgorithm::Fcfs);
        assert_eq!(sched.arm_position(), 100);
    }

    #[test]
    fn test_direction_setting() {
        let mut sched = DiskScheduler::new(200, 100);
        sched.set_direction(Direction::Down);
        assert_eq!(sched.direction(), Direction::Down);
    }

    #[test]
    fn test_compare_algorithms() {
        let results = compare_algorithms(200, 53, &sample_requests(), Direction::Up);
        assert_eq!(results.len(), 6);
        for result in &results {
            assert!(!result.visit_order.is_empty());
            assert!(result.total_seek > 0);
        }
    }

    #[test]
    fn test_avg_seek() {
        let mut sched = DiskScheduler::new(200, 0);
        sched.add_requests(&[50, 100]);
        let result = sched.schedule(DiskAlgorithm::Fcfs);
        // Seek from 0 to 50 = 50, from 50 to 100 = 50
        assert_eq!(result.total_seek, 100);
        assert!((result.avg_seek - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_max_seek() {
        let mut sched = DiskScheduler::new(200, 0);
        sched.add_requests(&[10, 180, 20]);
        let result = sched.schedule(DiskAlgorithm::Fcfs);
        assert_eq!(result.max_seek, 170); // 180 -> 20 = 160, or 10 -> 180 = 170
    }

    #[test]
    fn test_cumulative_stats() {
        let mut sched = DiskScheduler::new(200, 0);
        sched.add_request(100);
        sched.schedule(DiskAlgorithm::Fcfs);
        sched.add_request(50);
        sched.schedule(DiskAlgorithm::Fcfs);
        assert_eq!(sched.total_served(), 2);
        assert_eq!(sched.cumulative_seek(), 150); // 0->100 + 100->50
    }

    #[test]
    fn test_reset() {
        let mut sched = DiskScheduler::new(200, 50);
        sched.add_request(100);
        sched.schedule(DiskAlgorithm::Fcfs);
        sched.reset(0);
        assert_eq!(sched.arm_position(), 0);
        assert_eq!(sched.cumulative_seek(), 0);
        assert_eq!(sched.total_served(), 0);
    }

    #[test]
    fn test_scan_direction_down() {
        let mut sched = DiskScheduler::new(200, 100);
        sched.set_direction(Direction::Down);
        sched.add_requests(&[50, 150]);
        let result = sched.schedule(DiskAlgorithm::Look);
        // Going down first: should visit 50, then reverse and visit 150
        assert_eq!(result.visit_order[0], 50);
    }
}
