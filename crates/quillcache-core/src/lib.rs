//! Core contracts and node-local state for the QuillCache attention runtime.
//!
//! These modules share one release and change together. Engine adapters,
//! services, production pool integrations, and native kernels remain outside
//! this crate because they have separate deployment or toolchain boundaries.

#![forbid(unsafe_code)]

pub mod attention;
pub mod pool;
pub mod runtime;
pub mod scheduler;
pub mod transport;
pub mod types;
