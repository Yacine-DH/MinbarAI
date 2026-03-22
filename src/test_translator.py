"""
Interactive text translator - for testing translations without microphone
Type Arabic text and see instant German translation
"""
from translate import Translator

# Load translator
translator = Translator(from_lang="ar", to_lang="de")

print("=" * 60)
print("MinbarAI - Interactive Text Translator")
print("=" * 60)
print("\nType Arabic text and press Enter to see German translation")
print("Type 'quit' to exit\n")

while True:
    arabic = input("🇸🇦 Arabic: ").strip()
    
    if arabic.lower() == 'quit':
        print("Goodbye!")
        break
    
    if not arabic:
        continue
    
    german = translator.translate(arabic)
    print(f"🇩🇪 German: {german}\n")
