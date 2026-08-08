
//! ============================================================================
//! Module: Telemetry Transport (`telemetry.rs`)
//! ============================================================================
//! Handles framing and transport state (USB vs UDP) for outbound telemetry.
//! Physical socket/UART operations are delegated to `TelemetryCallback`.
//! ============================================================================

use core::sync::atomic::{AtomicU8, Ordering};
use crate::types::{TelemetryMode, TelemetryEvent, TelemetryCallback};

const SYNC_WORD: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];

static ACTIVE_MODE: AtomicU8 = AtomicU8::new(TelemetryMode::Usb as u8);
static mut TX_CALLBACK: Option<TelemetryCallback> = None;

// Internal cache for UDP configuration
static mut UDP_IP: [u8; 16] = [0; 16];
static mut UDP_IP_LEN: usize = 0;
static mut UDP_PORT: u16 = 0;

pub struct TelemetryTransport;

impl TelemetryTransport {
    /// Initializes the transport and sets up the hardware callback.
    pub fn init(server_ip: Option<&str>, port: u16, callback: TelemetryCallback) -> bool {
        unsafe { TX_CALLBACK = Some(callback); }

        if let Some(ip) = server_ip {
            if !ip.is_empty() && ip != "0.0.0.0" {
                return Self::set_udp(ip, port).is_ok();
            }
        }

        Self::set_usb();
        true
    }

    /// Switches the active transport mode to UDP Wireless.
    pub fn set_udp(ip: &str, port: u16) -> Result<(), ()> {
        if ip.is_empty() || port == 0 {
            return Err(());
        }

        let bytes = ip.as_bytes();
        let len = core::cmp::min(bytes.len(), 15);
        
        unsafe {
            UDP_IP[..len].copy_from_slice(&bytes[..len]);
            UDP_IP_LEN = len;
            UDP_PORT = port;
        }

        if let Some(cb) = unsafe { TX_CALLBACK } {
            let ip_str = unsafe { core::str::from_utf8_unchecked(&UDP_IP[..UDP_IP_LEN]) };
            
            // Tell the callback to create the socket
            if cb(TelemetryEvent::ConfigureUdp { ip: ip_str, port }) < 0 {
                // Socket creation failed in the callback, fallback to USB
                Self::set_usb();
                return Err(());
            }
        }

        ACTIVE_MODE.store(TelemetryMode::Udp as u8, Ordering::Release);
        Ok(())
    }

    /// Switches the active transport mode to USB Serial Cable.
    pub fn set_usb() {
        ACTIVE_MODE.store(TelemetryMode::Usb as u8, Ordering::Release);
        if let Some(cb) = unsafe { TX_CALLBACK } {
            cb(TelemetryEvent::TeardownUdp);
        }
    }

    /// Returns the current operational mode.
    pub fn get_mode() -> TelemetryMode {
        if ACTIVE_MODE.load(Ordering::Acquire) == TelemetryMode::Udp as u8 {
            TelemetryMode::Udp
        } else {
            TelemetryMode::Usb
        }
    }

    /// Frames and sends the payload to the registered transport callback.
    pub fn send(payload: &[u8]) -> i32 {
        if payload.is_empty() {
            return -1;
        }

        let cb = match unsafe { TX_CALLBACK } {
            Some(cb) => cb,
            None => return -1,
        };

        match Self::get_mode() {
            TelemetryMode::Usb => {
                // Write SYNC WORD
                cb(TelemetryEvent::SendUsb { data: &SYNC_WORD });
                
                // Write Frame Size (Native Endian, matching C's standard fwrite behavior)
                let size = payload.len() as u16;
                cb(TelemetryEvent::SendUsb { data: &size.to_ne_bytes() });
                
                // Write Payload (Returns written bytes to match C behavior)
                cb(TelemetryEvent::SendUsb { data: payload })
            }
            TelemetryMode::Udp => {
                cb(TelemetryEvent::SendUdp { data: payload })
            }
        }
    }

    /// Cleans up resources and resets to USB.
    pub fn deinit() {
        Self::set_usb();
    }
}
