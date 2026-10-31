// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// These are defaults: change them to match your actual amplifier wiring.
constexpr unsigned MIC_PIN = A0;       // Teensy 4.1 pin 14; biased analogue input
constexpr unsigned SPEAKER_PWM_PIN = 10; // filtered PWM -> amplifier input
constexpr unsigned RECORD_BUTTON_PIN = 2; // optional button from pin to GND
constexpr unsigned MIC_SAMPLE_RATE = 16000;
constexpr unsigned VOICE_SAMPLE_RATE = 8000;
constexpr unsigned MAX_RECORD_SECONDS = 30;
constexpr int MIC_DIGITAL_GAIN = 1;     // start at 1; analogue clipping cannot be repaired

// Leave off for the first speech test. Enable after copying the SAME export's
// ModelData.h and the kit's Runtime.h into this sketch, and student.tsd onto SD.
#define VOICE_ENABLE_SUMMARIZER 0

// Start with short, quiet-room dictation. Commands mode can be selected by
// typing !commands; !dictation restores unrestricted model-vocabulary decoding.
constexpr const char* VOICE_COMMAND_GRAMMAR =
    "hello; save note; cancel; read last note; repeat; summarise note";
