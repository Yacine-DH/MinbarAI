"""
List all available audio input devices
"""
import sounddevice as sd

print("\n" + "=" * 60)
print("Available Audio Input Devices")
print("=" * 60 + "\n")

devices = sd.query_devices()

input_devices = []
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        input_devices.append((i, device))
        marker = "← DEFAULT" if i == sd.default.device[0] else ""
        print(f"[{i}] {device['name']}")
        print(f"    Input channels: {device['max_input_channels']}")
        print(f"    Sample rate: {device['default_samplerate']}")
        print(f"    {marker}\n")

if input_devices:
    default_id = sd.default.device[0]
    print(f"\n✓ Default input device: #{default_id}")
    print(f"  Name: {devices[default_id]['name']}")
else:
    print("❌ No input devices found!")
