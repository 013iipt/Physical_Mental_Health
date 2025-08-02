# mental_health_analyzer.py

likert_scale = {
    "1": ("Strongly Disagree", 1),
    "2": ("Disagree", 2),
    "3": ("Neutral", 3),
    "4": ("Agree", 4),
    "5": ("Strongly Agree", 5)
}

# Questions
likert_questions = [
    "1. How are you feeling today?",
    "2. Did you sleep well last night?",
    "3. aRE YOU EXPERIENCING ANY PHYSICAL PAIN?",
    "4. Do you feel fit to work?",
    "5. Overall Physical Readiness "
]

extra_questions = [
    "6. Did you completed your certification course yet?",
    "7. Do you have any past physiacl incidant history  ?"
]

total_phy_score = 0

print("\n---  Questions  ---")
for i, question in enumerate(likert_questions, start=1):
    print(f"\n{question}")
    for key, (text, _) in likert_scale.items():
        print(f"{key}. {text}")
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in likert_scale:
            score = (likert_scale[choice][1] / 5) * 7
            total_phy_score += score
            break
        else:
            print("Invalid choice! Please enter a number between 1 and 5.")

print("\n--- Questions  ---")
for i, question in enumerate(extra_questions, start=6):
    print(f"\n{question}")
    answer = input("Answer (Yes/No): ").strip().lower()
    if answer in ["yes", "y", "1"]:
        total_phy_score += 15
    elif answer in ["no", "n", "0"]:
        total_phy_score += 0
    else:
        print("Invalid input, counted as No.")

print(f"\n✅ Your Total Score: {total_phy_score} / 65")










import os
import soundfile as sf
import numpy as np
from transformers import pipeline
import sentiment
import moderation
from datetime import datetime
import librosa
from scipy import signal
import statistics

class MentalHealthAnalyzer:
    def __init__(self):
        self.questions = [
            "How are you feeling today?",
            "Do you feel like working today?",
            "How was your day yesterday?",
            "Are you worried about anything?",
            "Are you ready to work today?"
        ]
        
        # Mental health indicators and their weights
        self.positive_indicators = {
            'joy': 5, 'gratitude': 4, 'optimism': 4, 'hope': 4, 'love': 3,
            'approval': 3, 'relief': 3, 'pride': 3, 'admiration': 2, 'excitement': 2,
            'amusement': 2, 'caring': 2, 'desire': 1, 'realization': 1, 'surprise': 1
        }
        
        self.negative_indicators = {
            'sadness': -5, 'anger': -5, 'fear': -4, 'grief': -4, 'disgust': -3,
            'disappointment': -3, 'disapproval': -3, 'annoyance': -2, 'remorse': -2,
            'nervousness': -2, 'embarrassment': -1, 'confusion': -1
        }
        
        self.neutral_indicators = {'neutral': 0, 'curiosity': 0}

    def analyze_voice_pitch(self, audio_file_path):
        """
        Analyze voice pitch characteristics for mental health indicators
        """
        print(f"🎵 Analyzing voice pitch from: {audio_file_path}")
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            
            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get the most prominent pitch at each time
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Filter out silence
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {
                    'mean_pitch': 0,
                    'pitch_variability': 0,
                    'pitch_stability': 0,
                    'voice_energy': 0,
                    'pitch_analysis': 'No voice detected'
                }
            
            # Calculate pitch statistics
            mean_pitch = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_range = np.max(pitch_values) - np.min(pitch_values)
            
            # Calculate voice energy
            rms = np.sqrt(np.mean(y**2))
            
            # Analyze pitch stability (lower std = more stable)
            pitch_stability = max(0, 10 - (pitch_std / 50))  # Scale to 0-10
            
            # Mental health indicators from pitch
            pitch_analysis = self.interpret_pitch_for_mental_health(mean_pitch, pitch_std, pitch_stability, rms)
            
            return {
                'mean_pitch': mean_pitch,
                'pitch_variability': pitch_std,
                'pitch_stability': pitch_stability,
                'voice_energy': rms,
                'pitch_range': pitch_range,
                'pitch_analysis': pitch_analysis
            }
            
        except Exception as e:
            print(f"❌ Error analyzing pitch: {e}")
            return {
                'mean_pitch': 0,
                'pitch_variability': 0,
                'pitch_stability': 0,
                'voice_energy': 0,
                'pitch_analysis': f'Error: {str(e)}'
            }

    def interpret_pitch_for_mental_health(self, mean_pitch, pitch_std, stability, energy):
        """
        Interpret pitch characteristics for mental health assessment
        """
        analysis = []
        
        # Pitch level analysis
        if mean_pitch > 200:  # High pitch
            analysis.append("High pitch detected - may indicate stress, anxiety, or excitement")
        elif mean_pitch < 100:  # Low pitch
            analysis.append("Low pitch detected - may indicate depression, fatigue, or calmness")
        else:  # Normal pitch
            analysis.append("Normal pitch range detected")
        
        # Pitch variability analysis
        if pitch_std > 50:
            analysis.append("High pitch variability - may indicate emotional instability or stress")
        elif pitch_std < 10:
            analysis.append("Low pitch variability - may indicate emotional flatness or depression")
        else:
            analysis.append("Normal pitch variability - indicates emotional stability")
        
        # Stability analysis
        if stability > 7:
            analysis.append("High voice stability - indicates emotional control and confidence")
        elif stability < 3:
            analysis.append("Low voice stability - may indicate nervousness or emotional distress")
        else:
            analysis.append("Moderate voice stability")
        
        # Energy analysis
        if energy > 0.1:
            analysis.append("High voice energy - indicates engagement and positive mood")
        elif energy < 0.01:
            analysis.append("Low voice energy - may indicate fatigue, depression, or disengagement")
        else:
            analysis.append("Moderate voice energy")
        
        return " | ".join(analysis)

    def calculate_mental_health_score(self, sentiment_data, emotion_data, pitch_data):
        """
        Calculate mental health score out of 35
        """
        score = 0
        max_score = 35
        
        # Sentiment-based scoring (20 points)
        sentiment_score = 0
        for emotion, value in sentiment_data.items():
            if emotion in self.positive_indicators:
                sentiment_score += (value / 100) * self.positive_indicators[emotion]
            elif emotion in self.negative_indicators:
                sentiment_score += (value / 100) * self.negative_indicators[emotion]
        
        # Normalize sentiment score to 0-20 range
        sentiment_score = max(0, min(20, sentiment_score + 10))
        score += sentiment_score
        
        # Emotion-based scoring (10 points)
        emotion_score = 0
        if emotion_data in self.positive_indicators:
            emotion_score = self.positive_indicators[emotion_data]
        elif emotion_data in self.negative_indicators:
            emotion_score = 10 + self.negative_indicators[emotion_data]  # Convert negative to positive scale
        
        score += emotion_score
        
        # Pitch-based scoring (5 points)
        pitch_score = 0
        if pitch_data['pitch_stability'] > 0:
            pitch_score = min(5, pitch_data['pitch_stability'] / 2)  # Scale stability to 5 points
        
        score += pitch_score
        
        # Ensure score is within 0-35 range
        score = max(0, min(35, score))
        
        return {
            'total_score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100,
            'sentiment_score': sentiment_score,
            'emotion_score': emotion_score,
            'pitch_score': pitch_score
        }

    def get_mental_health_assessment(self, score_data):
        """
        Provide mental health assessment based on score
        """
        percentage = score_data['percentage']
        
        if percentage >= 80:
            return {
                'status': 'Excellent',
                'assessment': 'The person appears to be in excellent mental health with positive emotional state.',
                'recommendation': 'Continue current positive practices and maintain healthy routines.',
                'color': '🟢'
            }
        elif percentage >= 60:
            return {
                'status': 'Good',
                'assessment': 'The person shows good mental health with mostly positive indicators.',
                'recommendation': 'Minor improvements possible through stress management and positive activities.',
                'color': '🟡'
            }
        elif percentage >= 40:
            return {
                'status': 'Moderate',
                'assessment': 'The person shows moderate mental health with some concerns.',
                'recommendation': 'Consider stress reduction techniques, social support, and professional consultation if needed.',
                'color': '🟠'
            }
        elif percentage >= 20:
            return {
                'status': 'Concerning',
                'assessment': 'The person shows concerning mental health indicators.',
                'recommendation': 'Strongly recommend professional mental health consultation and support.',
                'color': '🔴'
            }
        else:
            return {
                'status': 'Critical',
                'assessment': 'The person shows critical mental health indicators requiring immediate attention.',
                'recommendation': 'Immediate professional mental health intervention strongly recommended.',
                'color': '🚨'
            }

    def analyze_audio_for_mental_health(self, audio_file_path):
        """
        Comprehensive mental health analysis of audio file
        """
        print(f"🧠 Starting mental health analysis for: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            print(f"❌ Error: Audio file not found at {audio_file_path}")
            return None
        
        try:
            # Read audio file
            audio_data, sample_rate = sf.read(audio_file_path)
            print(f"📊 Audio info: {len(audio_data)} samples, {sample_rate} Hz")
            
            # Mock transcription (in real scenario, use speech recognition)
            mock_transcription = "How are you today? I am feeling good, thank you for asking."
            print(f"🗣️ Transcription: {mock_transcription}")
            
            # Analyze sentiment
            print("\n🧠 Analyzing sentiment...")
            sentiment_result = sentiment.analyze_sentiment(mock_transcription)
            
            # Analyze moderation
            print("\n🛡️ Analyzing content moderation...")
            moderation_result = moderation.analyze_moderation(mock_transcription)
            
            # Analyze emotion
            print("\n🎭 Analyzing emotion...")
            classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
            emotion_result = classifier(mock_transcription)
            
            # Extract top emotion
            top_emotion = "neutral"
            if emotion_result and len(emotion_result) > 0:
                if isinstance(emotion_result[0], list) and len(emotion_result[0]) > 0:
                    top_emotion = emotion_result[0][0].get("label", "neutral")
                elif isinstance(emotion_result[0], dict):
                    top_emotion = emotion_result[0].get("label", "neutral")
            
            print(f"🎭 Detected emotion: {top_emotion}")
            
            # Analyze voice pitch
            print("\n🎵 Analyzing voice pitch...")
            pitch_data = self.analyze_voice_pitch(audio_file_path)
            
            # Calculate mental health score
            print("\n📊 Calculating mental health score...")
            score_data = self.calculate_mental_health_score(sentiment_result, top_emotion, pitch_data)
            
            # Get assessment
            assessment = self.get_mental_health_assessment(score_data)
            
            # Create comprehensive result
            analysis_result = {
                "audio_file": audio_file_path,
                "audio_samples": len(audio_data),
                "sample_rate": sample_rate,
                "transcribed_text": mock_transcription,
                "emotion": top_emotion,
                "sentiment": sentiment_result,
                "moderation": moderation_result,
                "pitch_analysis": pitch_data,
                "mental_health_score": score_data,
                "mental_health_assessment": assessment
            }
            
            # Generate comprehensive report
            self.generate_mental_health_report(analysis_result)
            
            # Save to folder
            self.save_mental_health_report(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error in mental health analysis: {e}")
            return None

    def generate_mental_health_report(self, analysis_result):
        """
        Generate comprehensive mental health report
        """
        print("\n" + "="*80)
        print("🧠 MENTAL HEALTH ANALYSIS REPORT")
        print("="*80)
        
        print(f"🎵 Audio File: {analysis_result['audio_file']}")
        print(f"📊 Audio Info: {analysis_result['audio_samples']} samples, {analysis_result['sample_rate']} Hz")
        print(f"🗣️ Transcription: {analysis_result['transcribed_text']}")
        print(f"🎭 Primary Emotion: {analysis_result['emotion']}")
        
        # Mental Health Score
        score_data = analysis_result['mental_health_score']
        assessment = analysis_result['mental_health_assessment']
        
        print(f"\n📊 MENTAL HEALTH SCORE: {score_data['total_score']:.1f}/35 ({score_data['percentage']:.1f}%)")
        print(f"{assessment['color']} Status: {assessment['status']}")
        print(f"📝 Assessment: {assessment['assessment']}")
        print(f"💡 Recommendation: {assessment['recommendation']}")
        
        # Score breakdown
        print(f"\n📈 Score Breakdown:")
        print(f"   • Sentiment Score: {score_data['sentiment_score']:.1f}/20")
        print(f"   • Emotion Score: {score_data['emotion_score']:.1f}/10")
        print(f"   • Voice Pitch Score: {score_data['pitch_score']:.1f}/5")
        
        # Sentiment Analysis
        print(f"\n🧠 Sentiment Analysis:")
        sentiment_data = analysis_result.get('sentiment', {})
        if sentiment_data:
            top_sentiments = sorted(sentiment_data.items(), key=lambda x: x[1], reverse=True)[:8]
            for label, score in top_sentiments:
                if score > 1.0:
                    print(f"   • {label}: {score:.2f}")
        
        # Voice Pitch Analysis
        pitch_data = analysis_result['pitch_analysis']
        print(f"\n🎵 Voice Pitch Analysis:")
        print(f"   • Mean Pitch: {pitch_data['mean_pitch']:.1f} Hz")
        print(f"   • Pitch Variability: {pitch_data['pitch_variability']:.1f}")
        print(f"   • Voice Stability: {pitch_data['pitch_stability']:.1f}/10")
        print(f"   • Voice Energy: {pitch_data['voice_energy']:.4f}")
        print(f"   • Analysis: {pitch_data['pitch_analysis']}")
        
        # Moderation
        print(f"\n🛡️ Content Moderation:")
        moderation_data = analysis_result.get('moderation', {})
        if moderation_data:
            flagged_items = {k: v for k, v in moderation_data.items() if v > 0.1}
            if flagged_items:
                for label, score in flagged_items.items():
                    print(f"   ⚠️ {label}: {score:.4f}")
            else:
                print("   ✅ No moderation risks detected")
        
        print("\n" + "="*80)

    def save_mental_health_report(self, analysis_result):
        """
        Save mental health report to folder
        """
        # Create mental health reports folder
        reports_folder = "mental_health_reports"
        if not os.path.exists(reports_folder):
            os.makedirs(reports_folder)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.splitext(os.path.basename(analysis_result['audio_file']))[0]
        filename = f"mental_health_{audio_filename}_{timestamp}.txt"
        filepath = os.path.join(reports_folder, filename)
        
        # Prepare content
        content = []
        content.append("="*80)
        content.append("MENTAL HEALTH ANALYSIS REPORT")
        content.append("="*80)
        content.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"🎵 Audio File: {analysis_result['audio_file']}")
        content.append(f"📊 Audio Info: {analysis_result['audio_samples']} samples, {analysis_result['sample_rate']} Hz")
        content.append("")
        
        # Transcription
        content.append("🗣️ TRANSCRIPTION:")
        content.append("-" * 40)
        content.append(analysis_result['transcribed_text'])
        content.append("")
        
        # Mental Health Score
        score_data = analysis_result['mental_health_score']
        assessment = analysis_result['mental_health_assessment']
        
        content.append("📊 MENTAL HEALTH SCORE:")
        content.append("-" * 40)
        content.append(f"Total Score: {score_data['total_score']:.1f}/35 ({score_data['percentage']:.1f}%)")
        final_score= score_data['total_score'] + total_phy_score 
        print(f"\n✅ Your Total Score: {final_score} / 100 ")
        content.append(f"Status: {assessment['status']}")
        content.append(f"Assessment: {assessment['assessment']}")
        content.append(f"Recommendation: {assessment['recommendation']}")
        content.append("")
        
        # Score breakdown
        content.append("📈 Score Breakdown:")
        content.append(f"• Sentiment Score: {score_data['sentiment_score']:.1f}/20")
        content.append(f"• Emotion Score: {score_data['emotion_score']:.1f}/10")
        content.append(f"• Voice Pitch Score: {score_data['pitch_score']:.1f}/5")
        content.append("")
        
        # Sentiment
        content.append("🧠 SENTIMENT ANALYSIS:")
        content.append("-" * 40)
        sentiment_data = analysis_result.get('sentiment', {})
        if sentiment_data:
            top_sentiments = sorted(sentiment_data.items(), key=lambda x: x[1], reverse=True)[:10]
            for label, score in top_sentiments:
                if score > 1.0:
                    content.append(f"• {label}: {score:.2f}")
        content.append("")
        
        # Voice Pitch
        pitch_data = analysis_result['pitch_analysis']
        content.append("🎵 VOICE PITCH ANALYSIS:")
        content.append("-" * 40)
        content.append(f"Mean Pitch: {pitch_data['mean_pitch']:.1f} Hz")
        content.append(f"Pitch Variability: {pitch_data['pitch_variability']:.1f}")
        content.append(f"Voice Stability: {pitch_data['pitch_stability']:.1f}/10")
        content.append(f"Voice Energy: {pitch_data['voice_energy']:.4f}")
        content.append(f"Analysis: {pitch_data['pitch_analysis']}")
        content.append("")
        
        # Moderation
        content.append("🛡️ CONTENT MODERATION:")
        content.append("-" * 40)
        moderation_data = analysis_result.get('moderation', {})
        if moderation_data:
            flagged_items = {k: v for k, v in moderation_data.items() if v > 0.1}
            if flagged_items:
                for label, score in flagged_items.items():
                    content.append(f"⚠️ {label}: {score:.4f}")
            else:
                content.append("✅ No moderation risks detected")
        
        content.append("")
        content.append("="*80)
        
        # Save to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            print(f"💾 Mental health report saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MentalHealthAnalyzer()
    
    # Analyze the audio file
    audio_file = "_How are you today I.wav"
    result = analyzer.analyze_audio_for_mental_health(audio_file)
    
    if result:
        print("\n✅ Mental health analysis completed successfully!")
        
    else:
        print("\n❌ Mental health analysis failed!") 

