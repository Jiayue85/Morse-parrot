// SPDX-License-Identifier: GPL-3.0-or-later
// Teensy 4.1: offline 16 kHz ADC -> SD -> pretrained TinySTT -> formant TTS.
// First test: !say hello world. Then !record 5 or !wav TEST.WAV.
// MORSE: recognised speech is played as Morse (speaker + LED). Test with
// "!morse sos", then "!record 3" and speak. "!morsemode off" restores notes.
#include <Arduino.h>
#include <SD.h>
#include <malloc.h>
#include "VoiceConfig.h"
#include "AudioIO.h"
#include <AudioTools.h>
#include <AudioTools/FFT/AudioRealFFT.h>
#define TINYSTT_NO_USING_NAMESPACE
#include <TinySTT.h>
#include <TinySTT/data/tinystt_model.h>
#include <TinyTTSTools.h>
#include <ctype.h>

#if VOICE_ENABLE_SUMMARIZER
#if defined(__IMXRT1062__)
// GCC/Teensy can report a section-type conflict if generated global tables
// share .progmem with the speech libraries' function-local PSTR strings.
// Give only the student descriptors/tokenizer their own flash section.
#pragma push_macro("PROGMEM")
#undef PROGMEM
#define PROGMEM __attribute__((section(".progmem.sd_student")))
#include "Runtime.h"
#pragma pop_macro("PROGMEM")
#else
#include "Runtime.h"
#endif
class SummaryReader : public sd_student::Reader {
 public:
  File file;
  uint32_t length = 0;
  bool read(uint32_t offset, void* out, size_t count) override {
    return file.seek(offset) && file.read(static_cast<uint8_t*>(out), count) == int(count);
  }
  uint32_t size() const override { return length; }
};
SummaryReader summaryReader;
sd_student::WeightCache summaryCache(summaryReader);
DMAMEM sd_student::Runtime summaryRuntime(summaryCache);
bool summaryReady = false;
#endif

AudioRealFFT speechFFT;
tinystt::ModelContainer speechModel;
tinystt::STTPipeline recogniser(speechFFT, 64);
G2PDictionaryAndRulesModel pronunciation;
FormantVocoder speechSynth(VOICE_SAMPLE_RATE);
voice_io::SpeakerOutput speakerOutput;
TinyTTSTools speech(pronunciation, speechSynth, speakerOutput);

char command[1100], transcript[1024], pendingNote[1024];
size_t commandUsed = 0;
bool commandOverflow = false, speechReady = false, sdReady = false;
bool transcriptTruncated = false;
float recognitionConfidence = 0;
unsigned speechVolumePercent = 20;

class PCMBufferInput : public Stream {
 public:
  PCMBufferInput(const int16_t* samples, uint32_t count)
      : bytes(reinterpret_cast<const uint8_t*>(samples)), length(count * 2) {}
  int available() override { return int(length - position); }
  int read() override { return position < length ? bytes[position++] : -1; }
  int peek() override { return position < length ? bytes[position] : -1; }
  void flush() override {}
  size_t write(uint8_t) override { return 0; }
 private:
  const uint8_t* bytes;
  uint32_t length, position = 0;
};

void reportSpeaker(const char* tag, bool ok, uint32_t elapsed) {
  Serial.printf("%s ok=%u pin=%u elapsed_ms=%lu samples=%lu played=%lu peak_pcm=%lu pwm_min=%u pwm_max=%u underrun_samples=%lu volume_percent=%u source_peak_pcm=%lu\n",
                tag, unsigned(ok), SPEAKER_PWM_PIN, (unsigned long)elapsed,
                (unsigned long)voice_io::speakerQueued, (unsigned long)voice_io::speakerPlayed,
                (unsigned long)voice_io::speakerPeak, unsigned(voice_io::speakerPwmMin),
                unsigned(voice_io::speakerPwmMax), (unsigned long)voice_io::speakerUnderruns,
                speakerOutput.volumePercent(), (unsigned long)voice_io::speakerSourcePeak);
}

void testTone() {
  // A moderate-level 1 kHz sine, independent of the text synthesiser and SD.
  static const int16_t wave[8] = {0, 2896, 4096, 2896, 0, -2896, -4096, -2896};
  Serial.printf("TONE_TEST pin=%u frequency_hz=1000 duration_ms=750\n", SPEAKER_PWM_PIN);
  speakerOutput.setVolumePercent(100);
  if (!speakerOutput.begin()) { Serial.println("ERROR: speaker timer unavailable"); return; }
  const uint32_t started = millis();
  for (unsigned i = 0; i < 6000; ++i) {
    const int16_t sample = wave[i % 8];
    speakerOutput.write(reinterpret_cast<const uint8_t*>(&sample), sizeof(sample));
  }
  speakerOutput.end();
  reportSpeaker("TONE", true, millis() - started);
}

// The upstream TTS tokenizer drops digits. Expand them before synthesis so
// numeric dates/quantities are audibly retained (digit by digit).
bool spokenText(const char* input, char* out, size_t capacity) {
  static const char* digits[] = {" zero ", " one ", " two ", " three ", " four ", " five ", " six ", " seven ", " eight ", " nine "};
  size_t used = 0;
  while (*input) {
    char single[2] = {*input, 0};
    const char* replacement = single;
    if (*input >= '0' && *input <= '9') replacement = digits[*input - '0'];
    else if (*input == '-' && input[1] >= '0' && input[1] <= '9') replacement = " minus ";
    else if (*input == '+' && input[1] >= '0' && input[1] <= '9') replacement = " plus ";
    else if (*input == '/') replacement = " slash ";
    else if (*input == ':') replacement = " colon ";
    const size_t length = strlen(replacement);
    if (used + length >= capacity) return false;
    memcpy(out + used, replacement, length); used += length; ++input;
  }
  out[used] = 0;
  return true;
}

void speak(const char* text) {
  Serial.print("SPEAK "); Serial.println(text);
  static char normalisedSpeech[8192];
  if (!spokenText(text, normalisedSpeech, sizeof(normalisedSpeech))) { Serial.println("ERROR: spoken text too long"); return; }
  speakerOutput.setVolumePercent(speechVolumePercent);
  if (!speakerOutput.begin()) { Serial.println("ERROR: speaker timer unavailable"); return; }
  const uint32_t started = millis();
  const bool ok = speech.say(normalisedSpeech);
  speakerOutput.end();
  reportSpeaker("TTS", ok, millis() - started);
}

void onRecognition(const tinystt::SttResult& result, void*) {
  if (!result.is_final) return;
  recognitionConfidence = result.confidence();
  if (!result.accepted || !result.text()[0]) return;
  transcriptTruncated = strlen(result.text()) >= sizeof(transcript);
  strlcpy(transcript, result.text(), sizeof(transcript));
}

void reportMemory() {
  const auto heap = mallinfo();
  Serial.printf("MEMORY heap_allocated=%lu heap_free_blocks=%lu cpu_mhz=%u\n",
                (unsigned long)heap.uordblks, (unsigned long)heap.fordblks, unsigned(F_CPU_ACTUAL / 1000000));
}

void saveNote() {
  if (!pendingNote[0]) { speak("There is no note waiting to be saved."); return; }
  if (!sdReady) { speak("The memory card is unavailable."); return; }
  // Write a temporary file, close it, then rename. A damaged/interrupted .tmp
  // never replaces the previous completed note.
  if (SD.exists("note.tmp")) { speak("An unfinished note file needs checking."); return; }
  File file = SD.open("note.tmp", FILE_WRITE);
  if (!file) { speak("I could not save the note."); return; }
  const size_t length = strlen(pendingNote);
  const bool ok = file.write(reinterpret_cast<const uint8_t*>(pendingNote), length) == length;
  file.flush(); file.close();
  if (!ok) { speak("The note write failed."); return; }
  char target[24];
  bool renamed = false;
  for (unsigned i = 1; i <= 9999; ++i) {
    snprintf(target, sizeof(target), "note%04u.txt", i);
    if (!SD.exists(target)) { renamed = SD.rename("note.tmp", target); break; }
  }
  if (!renamed) { speak("The note could not be finalised."); return; }
  Serial.print("SAVED "); Serial.println(target);
  pendingNote[0] = 0;
  speak("Note saved.");
}

bool readLastNote(char* out, size_t capacity) {
  if (!sdReady) return false;
  File root = SD.open("/");
  if (!root) return false;
  unsigned latest = 0;
  for (File entry = root.openNextFile(); entry; entry = root.openNextFile()) {
    const char* name = entry.name();
    if (!entry.isDirectory() && strlen(name) == 12 && !strncasecmp(name, "note", 4) && !strcasecmp(name + 8, ".txt")) {
      bool digits = true;
      for (unsigned j = 4; j < 8; ++j) digits &= name[j] >= '0' && name[j] <= '9';
      const unsigned id = digits ? strtoul(name + 4, nullptr, 10) : 0;
      if (id > latest) latest = id;
    }
    entry.close();
  }
  root.close();
  if (!latest) return false;
  char path[24];
  snprintf(path, sizeof(path), "note%04u.txt", latest);
  File file = SD.open(path);
  if (!file || file.size() >= capacity) { if (file) file.close(); return false; }
  const size_t size = file.readBytes(out, capacity - 1);
  out[size] = 0; file.close();
  return size > 0;
}

void summariseNote() {
#if VOICE_ENABLE_SUMMARIZER
  if (!summaryReady) { speak("The summary model is unavailable."); return; }
  static char note[1024];
  if (pendingNote[0]) strlcpy(note, pendingNote, sizeof(note));
  else if (!readLastNote(note, sizeof(note))) { speak("There is no saved note."); return; }
  const uint32_t started = millis();
  const auto result = summaryRuntime.run(note);
  Serial.printf("SUMMARY_RESULT status=%s elapsed_ms=%lu\n", result.status, (unsigned long)(millis() - started));
  if (!strcmp(result.status, "ok")) speak(summaryRuntime.work.summary);
  else speak("I could not produce a complete summary.");
#else
  speak("The summary model has not been installed.");
#endif
}

// ---------------------------------------------------------------------------
// Morse output: recognised text -> dots/dashes on the speaker PWM pin + LED.
// Timing is driven by the speaker sample clock, so it stays accurate.
// ---------------------------------------------------------------------------
bool morseMode = true;                       // !morsemode off restores note behaviour
constexpr unsigned MORSE_UNIT_MS = 80;       // dot length; raise for slower Morse
constexpr float MORSE_TONE_HZ = 700.0f;
constexpr unsigned MORSE_VOLUME_PERCENT = 30;

const char* const MORSE_LETTERS[26] = {
  ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---",
  "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-",
  "..-", "...-", ".--", "-..-", "-.--", "--.."
};
const char* const MORSE_DIGITS[10] = {
  "-----", ".----", "..---", "...--", "....-",
  ".....", "-....", "--...", "---..", "----."
};

// Play (on=true) or silence (on=false) for ms milliseconds.
void morseChunk(bool on, uint32_t ms) {
  static float phase = 0;
  const uint32_t samples = ms * VOICE_SAMPLE_RATE / 1000;
  digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
  for (uint32_t i = 0; i < samples; ++i) {
    // Keep the queue ~20 ms ahead of playback so the LED stays in sync.
    while (voice_io::speakerQueued - voice_io::speakerPlayed > 160) yield();
    int16_t sample = 0;
    if (on) {
      sample = int16_t(12000.0f * sinf(phase));
      phase += 2.0f * PI * MORSE_TONE_HZ / VOICE_SAMPLE_RATE;
      if (phase > 2.0f * PI) phase -= 2.0f * PI;
    }
    speakerOutput.write(reinterpret_cast<const uint8_t*>(&sample), sizeof(sample));
  }
}

void playMorse(const char* text) {
  speakerOutput.setVolumePercent(MORSE_VOLUME_PERCENT);
  if (!speakerOutput.begin()) { Serial.println("ERROR: speaker timer unavailable"); return; }
  static char meaning[1100];                 // decoded text of what is played
  size_t meaningUsed = 0;
  meaning[0] = 0;
  Serial.print("MORSE ");
  for (const char* p = text; *p; ++p) {
    const char c = toupper(static_cast<unsigned char>(*p));
    if (c == ' ') {                          // letter gap (3) + 4 = word gap (7)
      Serial.print("/ ");
      if (meaningUsed && meaning[meaningUsed - 1] != ' ' && meaningUsed + 1 < sizeof(meaning)) {
        meaning[meaningUsed++] = ' '; meaning[meaningUsed] = 0;
      }
      morseChunk(false, MORSE_UNIT_MS * 4);
      continue;
    }
    const char* code = nullptr;
    if (c >= 'A' && c <= 'Z') code = MORSE_LETTERS[c - 'A'];
    else if (c >= '0' && c <= '9') code = MORSE_DIGITS[c - '0'];
    if (!code) continue;                     // skip anything with no Morse form
    Serial.printf("%s(%c) ", code, c);       // e.g. ....(H)
    if (meaningUsed + 1 < sizeof(meaning)) { meaning[meaningUsed++] = c; meaning[meaningUsed] = 0; }
    for (const char* s = code; *s; ++s) {
      morseChunk(true, *s == '.' ? MORSE_UNIT_MS : MORSE_UNIT_MS * 3);
      morseChunk(false, MORSE_UNIT_MS);      // gap between symbols
    }
    morseChunk(false, MORSE_UNIT_MS * 2);    // symbol gap (1) + 2 = letter gap (3)
  }
  Serial.println();
  Serial.printf("MEANING %s\n", meaning);
  speakerOutput.end();
  digitalWrite(LED_BUILTIN, LOW);
}

void handleTranscript() {
  if (transcriptTruncated) { speak("That recording produced too much text. Please try a shorter note."); return; }
  if (!transcript[0]) { speak("I could not recognise that. Please try again."); return; }
  Serial.printf("TRANSCRIPT confidence=%.3f text=%s\n", recognitionConfidence, transcript);
  if (morseMode) { playMorse(transcript); return; }
  char normalised[sizeof(transcript)];
  size_t used = 0;
  for (const char* p = transcript; *p && used + 1 < sizeof(normalised); ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    if (isalnum(c) || c == ' ') normalised[used++] = tolower(c);
  }
  while (used && normalised[used - 1] == ' ') --used;
  normalised[used] = 0;
  if (!strcmp(normalised, "save note")) saveNote();
  else if (!strcmp(normalised, "cancel")) { pendingNote[0] = 0; speak("Cancelled."); }
  else if (!strcmp(normalised, "hello")) speak("Hello. I am running offline on the Teensy.");
  else if (!strcmp(normalised, "read last note")) {
    static char note[1024];
    if (readLastNote(note, sizeof(note))) speak(note);
    else speak("There is no saved note.");
  } else if (!strcmp(normalised, "summarise note") || !strcmp(normalised, "summarize note")) summariseNote();
  else if (!strcmp(normalised, "repeat")) {
    if (pendingNote[0]) speak(pendingNote); else speak("There is no pending note.");
  } else {
    strlcpy(pendingNote, transcript, sizeof(pendingNote));
    speak("I heard."); speak(pendingNote);
    speak("Say save note to keep it, or cancel.");
  }
}

bool decodePCMStream(Stream& input, uint32_t bytes) {
  if (!speechReady || bytes % 2) return false;
  transcript[0] = 0; transcriptTruncated = false; recognitionConfidence = 0;
  auto& acoustic = recogniser.speechToPhonemes();
  acoustic.featureExtractor().reset();
  acoustic.acousticModel().reset();
  recogniser.phonemesToText().decoder().reset();
  const uint32_t droppedBefore = acoustic.droppedFrames();
  int16_t block[160];
  const uint32_t started = millis();
  Serial.printf("RECOGNISING audio_ms=%lu\n", (unsigned long)(bytes / 32));
  while (bytes) {
    const uint32_t take = bytes < sizeof(block) ? bytes : sizeof(block);
    memset(block, 0, sizeof(block));
    if (input.readBytes(reinterpret_cast<char*>(block), take) != take) return false;
    recogniser.write(reinterpret_cast<const uint8_t*>(block), sizeof(block));
    recogniser.copy();
    bytes -= take;
    yield();
  }
  // Padding flushes the FFT window, the final phoneme and the batched model.
  memset(block, 0, sizeof(block));
  for (unsigned i = 0; i < 30; ++i) {
    recogniser.write(reinterpret_cast<const uint8_t*>(block), sizeof(block));
    recogniser.copy();
  }
  recogniser.endOfUtterance(); recogniser.copy();
  const uint32_t dropped = acoustic.droppedFrames() - droppedBefore;
  Serial.printf("STT elapsed_ms=%lu dropped_frames=%lu\n", (unsigned long)(millis() - started), (unsigned long)dropped);
  reportMemory();
  return dropped == 0;
}

bool decodePCM(File& file, uint32_t offset, uint32_t bytes) {
  return file.seek(offset) && decodePCMStream(file, bytes);
}

void recordInRAM(unsigned seconds, bool useButton) {
  const uint32_t capacity = seconds * MIC_SAMPLE_RATE;
  int16_t* samples = static_cast<int16_t*>(malloc(capacity * sizeof(int16_t)));
  if (!samples) { Serial.println("ERROR: insufficient heap for RAM recording; use !record 1"); return; }
  Serial.printf("RECORDING RAM max_seconds=%u; transcript is unsaved until an SD card is mounted\n", seconds);
  digitalWrite(LED_BUILTIN, HIGH);
  if (!voice_io::beginCapture(samples, capacity)) {
    free(samples); digitalWrite(LED_BUILTIN, LOW); Serial.println("ERROR: ADC timer unavailable"); return;
  }
  while (voice_io::captureCount < capacity) {
    if (useButton && digitalRead(RECORD_BUTTON_PIN) == HIGH) break;
    yield();
  }
  voice_io::captureTimer.end();
  voice_io::captureDestination = nullptr;
  digitalWrite(LED_BUILTIN, LOW);
  const uint32_t count = voice_io::captureCount;
  Serial.printf("CAPTURE storage=RAM bytes=%lu adc_min=%u adc_max=%u clipped_samples=%lu\n",
                (unsigned long)(count * 2), voice_io::captureMin, voice_io::captureMax,
                (unsigned long)voice_io::captureClipped);
  if (count < MIC_SAMPLE_RATE / 2) { free(samples); speak("Please record for longer."); return; }
  PCMBufferInput input(samples, count);
  const bool ok = decodePCMStream(input, count * 2);
  free(samples);  // Release the recording before synthesis allocates its buffers.
  if (ok) handleTranscript(); else speak("Speech recognition failed.");
}

void recordAndRecognise(unsigned seconds, bool useButton) {
  if (!speechReady) { Serial.println("ERROR: recogniser not ready"); return; }
  seconds = constrain(seconds, 1u, MAX_RECORD_SECONDS);
  if (!sdReady) {
    if (seconds > voice_io::kMaxRamSeconds && !useButton) {
      Serial.println("ERROR: no SD card mounted; use !record 3 for a RAM-only speech test, or insert the card and send !sd"); return;
    }
    recordInRAM(min(seconds, voice_io::kMaxRamSeconds), useButton);
    return;
  }
  // Never overwrite a recording that the user might need for diagnosis.
  char path[24]; bool found = false;
  for (unsigned i = 1; i <= 9999; ++i) {
    snprintf(path, sizeof(path), "rec%04u.raw", i);
    if (!SD.exists(path)) { found = true; break; }
  }
  if (!found) { Serial.println("ERROR: recording filenames exhausted"); return; }
  File file = SD.open(path, FILE_WRITE);
  if (!file) { Serial.println("ERROR: cannot create recording"); return; }
  Serial.printf("RECORDING %s max_seconds=%u\n", path, seconds);
  digitalWrite(LED_BUILTIN, HIGH);
  if (!voice_io::beginCapture()) { file.close(); Serial.println("ERROR: ADC timer unavailable"); digitalWrite(LED_BUILTIN, LOW); return; }
  const uint32_t target = seconds * MIC_SAMPLE_RATE;
  bool writeOK = true;
  while (voice_io::captureCount < target) {
    if (useButton && digitalRead(RECORD_BUTTON_PIN) == HIGH) break;
    const uint32_t available = voice_io::captureWrite - voice_io::captureRead;
    if (available >= 512) {
      const uint32_t index = voice_io::captureRead % voice_io::kCaptureSize;
      if (file.write(reinterpret_cast<const uint8_t*>(voice_io::capture + index), 1024) != 1024) { writeOK = false; break; }
      voice_io::memoryBarrier(); voice_io::captureRead += 512;
    } else yield();
  }
  voice_io::captureTimer.end();
  digitalWrite(LED_BUILTIN, LOW);
  while (writeOK && voice_io::captureRead != voice_io::captureWrite) {
    const uint32_t index = voice_io::captureRead % voice_io::kCaptureSize;
    uint32_t n = voice_io::captureWrite - voice_io::captureRead;
    if (n > voice_io::kCaptureSize - index) n = voice_io::kCaptureSize - index;
    if (file.write(reinterpret_cast<const uint8_t*>(voice_io::capture + index), n * 2) != int(n * 2)) { writeOK = false; break; }
    voice_io::captureRead += n;
  }
  const uint32_t bytes = file.size();
  file.flush(); file.close();
  Serial.printf("CAPTURE bytes=%lu dropped_samples=%lu adc_min=%u adc_max=%u clipped_samples=%lu\n", (unsigned long)bytes,
                (unsigned long)voice_io::captureDropped, voice_io::captureMin, voice_io::captureMax, (unsigned long)voice_io::captureClipped);
  if (!writeOK || voice_io::captureDropped) { speak("The recording had an error. Please try again."); return; }
  if (bytes < MIC_SAMPLE_RATE) { speak("Please record for longer."); return; }
  file = SD.open(path, FILE_READ);
  const bool ok = file && decodePCM(file, 0, bytes);
  if (file) file.close();
  if (ok) handleTranscript(); else speak("Speech recognition failed.");
}

uint32_t little32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
void recogniseWav(const char* path) {
  File file = SD.open(path, FILE_READ);
  uint8_t header[16];
  if (!file || file.read(header, 12) != 12 || memcmp(header, "RIFF", 4) || memcmp(header + 8, "WAVE", 4)) {
    if (file) file.close(); Serial.println("ERROR: expected RIFF WAV"); return;
  }
  bool formatOK = false, decoded = false;
  while (file.available() >= 8) {
    if (file.read(header, 8) != 8) break;
    const uint32_t bytes = little32(header + 4), start = file.position();
    if (bytes > file.size() - start) break;
    if (!memcmp(header, "fmt ", 4) && bytes >= 16) {
      if (file.read(header, 16) != 16) break;
      formatOK = header[0] == 1 && header[1] == 0 && header[2] == 1 && header[3] == 0 &&
                 little32(header + 4) == MIC_SAMPLE_RATE && header[14] == 16 && header[15] == 0;
    } else if (!memcmp(header, "data", 4)) {
      if (!formatOK || bytes > MAX_RECORD_SECONDS * MIC_SAMPLE_RATE * 2) break;
      decoded = decodePCM(file, start, bytes); break;
    }
    if (!file.seek(start + bytes + (bytes & 1))) break;
  }
  file.close();
  if (decoded) handleTranscript(); else Serial.println("ERROR: need PCM WAV, mono, 16000 Hz, 16 bit, at most 30 s");
}

void executeCommand(char* text) {
  if (!strncmp(text, "!say ", 5)) speak(text + 5);
  else if (!strncmp(text, "!volume ", 8)) {
    char* end = nullptr; const long percent = strtol(text + 8, &end, 10);
    if (!end || end == text + 8 || *end || percent < 0 || percent > 100) Serial.println("ERROR: use !volume 0 through !volume 100");
    else { speechVolumePercent = unsigned(percent); Serial.printf("SPEECH_VOLUME percent=%u\n", speechVolumePercent); }
  }
  else if (!strncmp(text, "!morsemode ", 11)) {
    if (!strcmp(text + 11, "on")) morseMode = true;
    else if (!strcmp(text + 11, "off")) morseMode = false;
    else { Serial.println("ERROR: use !morsemode on or !morsemode off"); return; }
    Serial.printf("MORSE_MODE %s\n", morseMode ? "on" : "off");
  }
  else if (!strncmp(text, "!morse ", 7)) playMorse(text + 7);
  else if (!strcmp(text, "!tone")) testTone();
  else if (!strcmp(text, "!sd")) {
#if VOICE_ENABLE_SUMMARIZER
    summaryReader.file.close(); summaryReady = false;
#endif
    sdReady = SD.begin(BUILTIN_SDCARD);
    Serial.printf("SD_READY %u socket=built_in\n", unsigned(sdReady));
#if VOICE_ENABLE_SUMMARIZER
    if (sdReady) {
      summaryReader.file = SD.open("student.tsd", FILE_READ);
      summaryReader.length = summaryReader.file ? summaryReader.file.size() : 0;
      summaryReady = summaryReader.file && summaryCache.verify();
      Serial.printf("SUMMARY_READY %u\n", unsigned(summaryReady));
    }
#endif
    if (!sdReady) Serial.println("Short recording is available with !record 3; note saving still requires SD.");
  }
  else if (!strncmp(text, "!record ", 8)) {
    char* end = nullptr; const long seconds = strtol(text + 8, &end, 10);
    if (!end || *end || seconds < 1 || seconds > MAX_RECORD_SECONDS) Serial.println("ERROR: use !record 1 through !record 30");
    else recordAndRecognise(unsigned(seconds), false);
  } else if (!strncmp(text, "!wav ", 5)) recogniseWav(text + 5);
  else if (!strcmp(text, "!commands")) {
    Serial.println(recogniser.phonemesToText().setGrammar(VOICE_COMMAND_GRAMMAR) ? "MODE commands" : "ERROR: grammar");
  } else if (!strcmp(text, "!dictation")) { recogniser.phonemesToText().clearGrammar(); Serial.println("MODE dictation"); }
  else if (!strcmp(text, "!save")) saveNote();
  else if (!strcmp(text, "!cancel")) { pendingNote[0] = 0; speak("Cancelled."); }
  else if (!strcmp(text, "!summary")) summariseNote();
  else if (!strcmp(text, "!read")) {
    static char note[1024]; if (readLastNote(note, sizeof(note))) speak(note); else speak("There is no saved note.");
  } else if (!strcmp(text, "!info")) reportMemory();
  else Serial.println("Commands: !morse TEXT | !morsemode on/off | !tone | !say TEXT | !volume 20 | !record 3 | !sd | !wav TEST.WAV | !commands | !dictation | !save | !cancel | !read | !summary | !info");
}

void setup() {
  Serial.begin(115200);
  const uint32_t start = millis(); while (!Serial && millis() - start < 3000) {}
  pinMode(LED_BUILTIN, OUTPUT); pinMode(RECORD_BUTTON_PIN, INPUT_PULLUP);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);
  Serial.println("TEENSY_VOICE offline pretrained STT + formant TTS; experimental Teensy port");
  speech.begin(VOICE_SAMPLE_RATE);
  sdReady = SD.begin(BUILTIN_SDCARD);
  Serial.printf("SD_READY %u socket=built_in\n", unsigned(sdReady));
  if (!sdReady) Serial.println("No SD card mounted. Use !record 3 for a RAM-only speech test, or insert a card and send !sd.");
  // A .progmem section is required on Teensy; default .rodata is copied to RAM.
  const uintptr_t modelAddress = reinterpret_cast<uintptr_t>(tinystt_model);
  if (modelAddress < 0x60000000u || modelAddress >= 0x60800000u) {
    Serial.println("ERROR: STT model is not in program flash; install the supplied patched libraries"); return;
  }
  const auto status = speechModel.begin(tinystt_model, tinystt_model_size());
  Serial.printf("STT_MODEL bytes=%u address=0x%08lx container_status=%u\n", unsigned(tinystt_model_size()), (unsigned long)modelAddress, unsigned(status));
  if (status != tinystt::ModelContainer::OK) return;
  recogniser.speechToPhonemes().setVADEnabled(false);
  recogniser.speechToPhonemes().setPreroll(0);
  recogniser.speechToPhonemes().setBatchFrames(4);
  recogniser.phonemesToText().decoderConfig().partial_results = false;
  recogniser.phonemesToText().decoderConfig().nbest = 1;
  recogniser.phonemesToText().setResultCallback(onRecognition);
  speechReady = recogniser.begin(speechModel);
  Serial.printf("STT_READY %u\n", unsigned(speechReady));
#if VOICE_ENABLE_SUMMARIZER
  if (sdReady) {
    summaryReader.file = SD.open("student.tsd", FILE_READ);
    summaryReader.length = summaryReader.file ? summaryReader.file.size() : 0;
    summaryReady = summaryReader.file && summaryCache.verify();
    Serial.printf("SUMMARY_READY %u arena_bytes=%u\n", unsigned(summaryReady), unsigned(sizeof(summaryRuntime) + sizeof(summaryCache)));
  }
#endif
  reportMemory();
  Serial.printf("AUDIO_PINS mic=%u speaker_pwm=%u button=%u\n", MIC_PIN, SPEAKER_PWM_PIN, RECORD_BUTTON_PIN);
  Serial.printf("SPEECH_VOLUME percent=%u\n", speechVolumePercent);
  Serial.println("Try !tone, !say hello world, then !record 3. Speak while the LED is on. Or hold the optional record button.");
  speak("Ready for a speech test.");
}

void loop() {
  while (Serial.available()) {
    const char c = Serial.read(); if (c == '\r') continue;
    if (c != '\n') {
      if (commandUsed + 1 < sizeof(command)) command[commandUsed++] = c; else commandOverflow = true;
      continue;
    }
    command[commandUsed] = 0;
    if (commandOverflow) Serial.println("ERROR: command too long");
    else if (commandUsed) executeCommand(command);
    commandUsed = 0; commandOverflow = false;
  }
  if (digitalRead(RECORD_BUTTON_PIN) == LOW) {
    delay(25);
    if (digitalRead(RECORD_BUTTON_PIN) == LOW) {
      recordAndRecognise(MAX_RECORD_SECONDS, true);
      while (digitalRead(RECORD_BUTTON_PIN) == LOW) delay(10);
    }
  }
}
