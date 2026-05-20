//! 24/7 Carbon-Free Energy (CFE) matching and scheduling.
//!
//! Provides hourly CFE profiles and scheduling decisions to maximize
//! the use of carbon-free energy sources for workload execution.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CfeSource
// ---------------------------------------------------------------------------

/// The type of carbon-free (or grid) energy source for a given hour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CfeSource {
    /// Photovoltaic / solar energy.
    Solar,
    /// Wind turbine energy.
    Wind,
    /// Hydroelectric energy.
    Hydro,
    /// Nuclear energy.
    Nuclear,
    /// Conventional grid (may include fossil fuels).
    Grid,
}

impl fmt::Display for CfeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solar => write!(f, "Solar"),
            Self::Wind => write!(f, "Wind"),
            Self::Hydro => write!(f, "Hydro"),
            Self::Nuclear => write!(f, "Nuclear"),
            Self::Grid => write!(f, "Grid"),
        }
    }
}

// ---------------------------------------------------------------------------
// CfeHour
// ---------------------------------------------------------------------------

/// Carbon-free energy availability for a single hour of the day.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfeHour {
    /// Hour of the day (0-23).
    pub hour: u8,
    /// Percentage of energy that is carbon-free (0.0 - 100.0).
    pub cfe_pct: f64,
    /// Dominant energy source for this hour.
    pub source: CfeSource,
    /// Available power from this source (megawatts).
    pub available_mw: f64,
}

// ---------------------------------------------------------------------------
// CfeSchedule
// ---------------------------------------------------------------------------

/// A 24-hour carbon-free energy schedule for a specific region and date.
///
/// Contains up to 24 [`CfeHour`] entries, one per hour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfeSchedule {
    /// Hourly CFE entries (up to 24).
    pub hours: Vec<CfeHour>,
    /// Region this schedule applies to.
    pub region: String,
    /// Date this schedule applies to (e.g. "2025-06-15").
    pub date: String,
}

impl CfeSchedule {
    /// Creates a new schedule for the given region and date with no hourly
    /// entries.
    pub fn new(region: &str, date: &str) -> Self {
        Self {
            hours: Vec::new(),
            region: region.to_string(),
            date: date.to_string(),
        }
    }

    /// Inserts or replaces the entry for a given hour.
    ///
    /// If an entry already exists for `hour.hour`, it is replaced.
    pub fn set_hour(&mut self, hour: CfeHour) {
        if let Some(existing) = self.hours.iter_mut().find(|h| h.hour == hour.hour) {
            *existing = hour;
        } else {
            self.hours.push(hour);
        }
    }

    /// Returns the entry for the given hour, if present.
    pub fn get_hour(&self, hour: u8) -> Option<&CfeHour> {
        self.hours.iter().find(|h| h.hour == hour)
    }

    /// Finds the best starting hour for a workload of `duration_hours`
    /// consecutive hours, maximising the average CFE percentage.
    ///
    /// Hours wrap around at 24.  Returns `None` if the schedule has no
    /// hourly entries.
    pub fn best_hour_for_workload(&self, duration_hours: u8) -> Option<u8> {
        if self.hours.is_empty() || duration_hours == 0 {
            return None;
        }

        let mut best_start: Option<u8> = None;
        let mut best_avg: f64 = f64::MIN;

        for start in 0u8..24 {
            let mut total_cfe = 0.0;
            let mut count = 0u8;

            for offset in 0..duration_hours {
                let h = (start + offset) % 24;
                if let Some(entry) = self.get_hour(h) {
                    total_cfe += entry.cfe_pct;
                    count += 1;
                }
            }

            if count > 0 {
                let avg = total_cfe / f64::from(count);
                if avg > best_avg {
                    best_avg = avg;
                    best_start = Some(start);
                }
            }
        }

        best_start
    }

    /// Determines whether a workload should be deferred to a higher-CFE hour.
    ///
    /// If the current hour already meets `min_cfe_pct`, returns `None`
    /// (run now).  Otherwise returns the next hour (searching forward,
    /// wrapping around) that meets the threshold.  Returns `None` if no
    /// qualifying hour exists.
    pub fn defer_for_cfe(&self, current_hour: u8, min_cfe_pct: f64) -> Option<u8> {
        // Check if the current hour already qualifies.
        if let Some(entry) = self.get_hour(current_hour)
            && entry.cfe_pct >= min_cfe_pct
        {
            return None; // Run now.
        }

        // Search forward through the remaining hours, then wrap around.
        for offset in 1u8..24 {
            let h = (current_hour + offset) % 24;
            if let Some(entry) = self.get_hour(h)
                && entry.cfe_pct >= min_cfe_pct
            {
                return Some(h);
            }
        }

        None
    }

    /// Average CFE percentage across all hours in the schedule.
    ///
    /// Returns `0.0` if the schedule has no entries.
    pub fn annual_cfe_score(&self) -> f64 {
        if self.hours.is_empty() {
            return 0.0;
        }
        let total: f64 = self.hours.iter().map(|h| h.cfe_pct).sum();
        total / self.hours.len() as f64
    }

    /// Returns the hour with the highest CFE percentage.
    pub fn peak_cfe_hour(&self) -> Option<&CfeHour> {
        self.hours.iter().max_by(|a, b| {
            a.cfe_pct
                .partial_cmp(&b.cfe_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Sum of `available_mw` across all non-Grid sources.
    pub fn total_renewable_mw(&self) -> f64 {
        self.hours
            .iter()
            .filter(|h| h.source != CfeSource::Grid)
            .map(|h| h.available_mw)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hour(hour: u8, cfe_pct: f64, source: CfeSource, mw: f64) -> CfeHour {
        CfeHour {
            hour,
            cfe_pct,
            source,
            available_mw: mw,
        }
    }

    #[test]
    fn create_schedule() {
        let schedule = CfeSchedule::new("eu-west-1", "2025-06-15");
        assert_eq!(schedule.region, "eu-west-1");
        assert_eq!(schedule.date, "2025-06-15");
        assert!(schedule.hours.is_empty());
    }

    #[test]
    fn set_and_get_hour() {
        let mut schedule = CfeSchedule::new("us-east-1", "2025-07-01");
        let h = make_hour(10, 85.0, CfeSource::Solar, 50.0);
        schedule.set_hour(h);

        let entry = schedule.get_hour(10).expect("hour 10 should exist");
        assert_eq!(entry.hour, 10);
        assert!((entry.cfe_pct - 85.0).abs() < 1e-10);

        // Upsert the same hour.
        schedule.set_hour(make_hour(10, 90.0, CfeSource::Solar, 55.0));
        let updated = schedule.get_hour(10).unwrap();
        assert!((updated.cfe_pct - 90.0).abs() < 1e-10);
    }

    #[test]
    fn best_hour_single_hour_workload() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(8, 40.0, CfeSource::Grid, 10.0));
        schedule.set_hour(make_hour(12, 95.0, CfeSource::Solar, 80.0));
        schedule.set_hour(make_hour(20, 60.0, CfeSource::Wind, 30.0));

        let best = schedule.best_hour_for_workload(1).unwrap();
        assert_eq!(best, 12);
    }

    #[test]
    fn best_hour_multi_hour_workload() {
        let mut schedule = CfeSchedule::new("r", "d");
        // Populate all 24 hours with a low baseline.
        for h in 0..24u8 {
            schedule.set_hour(make_hour(h, 10.0, CfeSource::Grid, 5.0));
        }
        // Override hours 10-12 with high CFE.
        schedule.set_hour(make_hour(10, 90.0, CfeSource::Solar, 50.0));
        schedule.set_hour(make_hour(11, 95.0, CfeSource::Solar, 55.0));
        schedule.set_hour(make_hour(12, 85.0, CfeSource::Solar, 45.0));
        // Override hours 18-20 with mediocre CFE.
        schedule.set_hour(make_hour(18, 30.0, CfeSource::Grid, 10.0));
        schedule.set_hour(make_hour(19, 35.0, CfeSource::Wind, 15.0));
        schedule.set_hour(make_hour(20, 25.0, CfeSource::Grid, 8.0));

        let best = schedule.best_hour_for_workload(3).unwrap();
        assert_eq!(best, 10);
    }

    #[test]
    fn best_hour_empty_schedule() {
        let schedule = CfeSchedule::new("r", "d");
        assert!(schedule.best_hour_for_workload(1).is_none());
    }

    #[test]
    fn defer_run_now() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(14, 80.0, CfeSource::Solar, 50.0));

        // Current hour meets threshold — run now (None).
        assert!(schedule.defer_for_cfe(14, 70.0).is_none());
    }

    #[test]
    fn defer_to_later_hour() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(8, 20.0, CfeSource::Grid, 10.0));
        schedule.set_hour(make_hour(12, 90.0, CfeSource::Solar, 60.0));

        let deferred = schedule.defer_for_cfe(8, 50.0);
        assert_eq!(deferred, Some(12));
    }

    #[test]
    fn defer_wraps_around() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(6, 85.0, CfeSource::Solar, 40.0));
        schedule.set_hour(make_hour(22, 20.0, CfeSource::Grid, 5.0));

        // Current hour is 22 (low), threshold 50.  Must wrap to hour 6.
        let deferred = schedule.defer_for_cfe(22, 50.0);
        assert_eq!(deferred, Some(6));
    }

    #[test]
    fn defer_no_qualifying_hour() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(0, 10.0, CfeSource::Grid, 5.0));
        schedule.set_hour(make_hour(12, 20.0, CfeSource::Grid, 8.0));

        // No hour meets the 90% threshold.
        assert!(schedule.defer_for_cfe(0, 90.0).is_none());
    }

    #[test]
    fn annual_cfe_score() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(0, 40.0, CfeSource::Grid, 10.0));
        schedule.set_hour(make_hour(12, 80.0, CfeSource::Solar, 50.0));

        // Average = (40 + 80) / 2 = 60.0
        assert!((schedule.annual_cfe_score() - 60.0).abs() < 1e-10);
    }

    #[test]
    fn annual_cfe_score_empty() {
        let schedule = CfeSchedule::new("r", "d");
        assert!((schedule.annual_cfe_score()).abs() < 1e-10);
    }

    #[test]
    fn peak_cfe_hour() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(6, 50.0, CfeSource::Wind, 20.0));
        schedule.set_hour(make_hour(13, 98.0, CfeSource::Solar, 70.0));
        schedule.set_hour(make_hour(22, 30.0, CfeSource::Grid, 10.0));

        let peak = schedule.peak_cfe_hour().expect("should have a peak");
        assert_eq!(peak.hour, 13);
        assert!((peak.cfe_pct - 98.0).abs() < 1e-10);
    }

    #[test]
    fn total_renewable_mw() {
        let mut schedule = CfeSchedule::new("r", "d");
        schedule.set_hour(make_hour(8, 80.0, CfeSource::Solar, 50.0));
        schedule.set_hour(make_hour(12, 70.0, CfeSource::Wind, 30.0));
        schedule.set_hour(make_hour(18, 20.0, CfeSource::Grid, 100.0));
        schedule.set_hour(make_hour(0, 90.0, CfeSource::Nuclear, 40.0));

        // Solar(50) + Wind(30) + Nuclear(40) = 120; Grid excluded.
        assert!((schedule.total_renewable_mw() - 120.0).abs() < 1e-10);
    }

    #[test]
    fn cfe_source_display() {
        assert_eq!(format!("{}", CfeSource::Solar), "Solar");
        assert_eq!(format!("{}", CfeSource::Wind), "Wind");
        assert_eq!(format!("{}", CfeSource::Hydro), "Hydro");
        assert_eq!(format!("{}", CfeSource::Nuclear), "Nuclear");
        assert_eq!(format!("{}", CfeSource::Grid), "Grid");
    }

    #[test]
    fn schedule_serialization() {
        let mut schedule = CfeSchedule::new("eu-north-1", "2025-12-01");
        schedule.set_hour(make_hour(10, 75.0, CfeSource::Hydro, 200.0));

        let json = serde_json::to_string(&schedule).expect("serialize");
        let d: CfeSchedule = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(d.region, "eu-north-1");
        assert_eq!(d.date, "2025-12-01");
        assert_eq!(d.hours.len(), 1);
        assert_eq!(d.hours[0].hour, 10);
    }

    #[test]
    fn hour_serialization() {
        let h = make_hour(15, 62.5, CfeSource::Wind, 35.0);
        let json = serde_json::to_string(&h).expect("serialize");
        let d: CfeHour = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(d.hour, 15);
        assert!((d.cfe_pct - 62.5).abs() < 1e-10);
        assert_eq!(d.source, CfeSource::Wind);
        assert!((d.available_mw - 35.0).abs() < 1e-10);
    }
}
