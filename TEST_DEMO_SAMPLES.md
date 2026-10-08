# Demo Samples from Training Data

## Tamil Depressed Samples

### Sample 1 (D_A00_6-1)
```
எதுவும் செய்ய முடில்லை. நான் ஒரு தோல் வி மாதிரி இருக்கிறேன். எல்லாரு என்ன விட்டுவிட்டு போய்தான் வேண்டும்.
```
**Translation**: "I can't do anything. I'm like a failure. Everyone should leave me and go."
**Label**: Depressed (1)

### Sample 2 (D_F2003)
```
கண்ணகூட முடியல் யாரும் என்ன பார்த்துக்கொண்டு இருக்கும் மாறிக் கொடுக்கும்.
```
**Label**: Depressed (1)

### Sample 3 (D_S00_54-4)
```
நான் நான்து நான் விலைப் போலும் வேறை யாரும் அரியும் நல்லந்திட்டுக்கொண்டிருக்கிறேன்.
```
**Label**: Depressed (1)

## Tamil Non-Depressed Sample

### Sample (ND1_0001)
```
பயமுருத்தலை விடனாய வஞ்சிகம் பலமானது.
```
**Label**: Non-Depressed (0)

## Malayalam Samples

### Depressed (D_A012_3)
```
وہ سپٹ்டப் போற்று உறுக்குப் போல மின்னிக்குத் தோனே இருந்துவிட்டில்லை.
```
**Label**: Depressed (1)

### Non-Depressed (ND1_0001)
```
ஒடத்தொடுத்த, ஒரு கிராமத்தில் அம்மா வசின் பேராயும் ஒருத்தனிக் கொண்டே இருக்கிறது.
```
**Label**: Non-Depressed (0)

## Usage in Demo

All samples are from the actual training dataset used to train the models. This ensures:

1. **Authentic Data** - Real transcripts from the competition dataset
2. **Known Labels** - Ground truth labels for validation
3. **Model Performance** - Shows how models perform on actual training data
4. **Traceability** - Each sample includes source file ID

## Expected Model Behavior

When testing with these samples, you should see:
- Models generally agreeing with the ground truth labels
- Confidence scores reflecting training performance
- Consensus predictions matching expected outcomes

Perfect for demonstrating model capabilities! 🎯
