// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <Arduino.h>
#include <IntervalTimer.h>
#include "VoiceConfig.h"

// One ADC producer and one SD writer. Record first, recognise afterwards:
// real-time recognition is deliberately not assumed on an unmeasured board.
namespace voice_io {
constexpr uint32_t kCaptureSize = 8192;
constexpr uint32_t kSpeakerSize = 2048;
constexpr unsigned kMaxRamSeconds = 3;
DMAMEM int16_t capture[kCaptureSize];
DMAMEM int16_t speaker[kSpeakerSize];
volatile uint32_t captureWrite = 0, captureRead = 0, captureDropped = 0;
volatile uint32_t captureCount = 0;
volatile uint16_t captureMin = 4095, captureMax = 0;
volatile uint32_t captureClipped = 0;
volatile uint32_t speakerWrite = 0, speakerRead = 0;
volatile uint32_t speakerUnderruns = 0;
volatile bool speakerHasStarted = false;
volatile uint32_t speakerPlayed = 0;
uint32_t speakerQueued = 0, speakerPeak = 0;
uint32_t speakerSourcePeak = 0;
uint8_t speakerPwmMin = 255, speakerPwmMax = 0;
int16_t* captureDestination = nullptr;
uint32_t captureCapacity = 0;
int32_t dcEstimate = 2048 << 12;
IntervalTimer captureTimer, speakerTimer;
inline void memoryBarrier() { __asm__ volatile("dmb" ::: "memory"); }

void sampleMic() {
  if (captureDestination && captureCount >= captureCapacity) return;
  const uint16_t value = analogRead(MIC_PIN);
  if (value < captureMin) captureMin = value;
  if (value > captureMax) captureMax = value;
  if (value < 8 || value > 4087) captureClipped++;
  // Slow DC tracking removes the amplifier's bias without removing speech.
  dcEstimate += ((int32_t(value) << 12) - dcEstimate) >> 12;
  int32_t sample = ((int32_t(value) << 12) - dcEstimate) >> 8;
  sample *= MIC_DIGITAL_GAIN;
  sample = sample < -32768 ? -32768 : (sample > 32767 ? 32767 : sample);
  if (captureDestination) {
    captureDestination[captureCount] = int16_t(sample);
    memoryBarrier();
  } else if (captureWrite - captureRead < kCaptureSize) {
    capture[captureWrite % kCaptureSize] = int16_t(sample);
    memoryBarrier();
    captureWrite++;
  } else captureDropped++;
  captureCount++;
}

void playSample() {
  uint8_t level = 128;
  if (speakerRead != speakerWrite) {
    level = (int32_t(speaker[speakerRead % kSpeakerSize]) + 32768) >> 8;
    memoryBarrier();
    speakerRead++;
    speakerPlayed++;
    speakerHasStarted = true;
  } else if (speakerHasStarted) speakerUnderruns++;
  analogWrite(SPEAKER_PWM_PIN, level);
}

bool beginCapture(int16_t* destination = nullptr, uint32_t capacity = 0) {
  if (destination && !capacity) return false;
  captureDestination = destination;
  captureCapacity = capacity;
  captureWrite = captureRead = captureDropped = captureCount = captureClipped = 0;
  captureMin = 4095; captureMax = 0;
  analogReadResolution(12);
  analogReadAveraging(1);
  dcEstimate = int32_t(analogRead(MIC_PIN)) << 12;
  return captureTimer.begin(sampleMic, 1000000.0f / MIC_SAMPLE_RATE);
}

class SpeakerOutput : public Print {
 public:
  using Print::write;
  void setVolumePercent(unsigned percent) { playbackVolumePercent = percent > 100 ? 100 : percent; }
  unsigned volumePercent() const { return playbackVolumePercent; }
  bool begin() {
    pinMode(SPEAKER_PWM_PIN, OUTPUT);
    analogWriteResolution(8);
    analogWriteFrequency(SPEAKER_PWM_PIN, 187500);
    analogWrite(SPEAKER_PWM_PIN, 128);
    speakerRead = speakerWrite = speakerUnderruns = 0;
    speakerHasStarted = false;
    speakerPlayed = speakerQueued = speakerPeak = speakerSourcePeak = 0;
    speakerPwmMin = 255; speakerPwmMax = 0;
    pending = false;
    active = speakerTimer.begin(playSample, 1000000.0f / VOICE_SAMPLE_RATE);
    return active;
  }
  size_t write(uint8_t byte) override {
    if (!active) return 0;
    if (!pending) { low = byte; pending = true; }
    else {
      const int16_t source = int16_t(uint16_t(low) | (uint16_t(byte) << 8));
      const uint32_t sourceMagnitude = source < 0 ? -int32_t(source) : int32_t(source);
      if (sourceMagnitude > speakerSourcePeak) speakerSourcePeak = sourceMagnitude;
      const int16_t sample = int32_t(source) * int32_t(playbackVolumePercent) / 100;
      const uint32_t magnitude = sample < 0 ? -int32_t(sample) : int32_t(sample);
      if (magnitude > speakerPeak) speakerPeak = magnitude;
      const uint8_t level = (int32_t(sample) + 32768) >> 8;
      if (level < speakerPwmMin) speakerPwmMin = level;
      if (level > speakerPwmMax) speakerPwmMax = level;
      while (speakerWrite - speakerRead >= kSpeakerSize) yield();
      speaker[speakerWrite % kSpeakerSize] = sample;
      memoryBarrier();
      speakerWrite++;
      speakerQueued++;
      pending = false;
    }
    return 1;
  }
  size_t write(const uint8_t* bytes, size_t length) override {
    if (!active) return 0;
    for (size_t i = 0; i < length; ++i) write(bytes[i]);
    return length;
  }
  void end() {
    if (!active) return;
    while (speakerRead != speakerWrite) yield();
    speakerTimer.end();
    analogWrite(SPEAKER_PWM_PIN, 128);
    active = false;
  }
 private:
  bool active = false, pending = false;
  uint8_t low = 0;
  unsigned playbackVolumePercent = 100;
};
}  // namespace voice_io
