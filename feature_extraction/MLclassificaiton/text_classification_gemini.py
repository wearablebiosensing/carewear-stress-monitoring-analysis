import google.generativeai as genai
import os

# Configure your Gemini API key from environment variables
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def classify_stress_with_gemini(prompt_text):
    """
    Uses the Gemini model to classify and justify a stress state.

    Args:
        prompt_text (str): The narrative text to be classified.

    Returns:
        str: The Gemini model's classification and justification.
    """
    # Create the full prompt including the classification instruction
    full_prompt = (
        "You are an expert on human physiological responses to stress. "
        "Analyze the following data to determine if the stress is physical or mental. "
        "Provide a clear classification and a brief justification.\n\n"
        f"Data: {prompt_text}\n\n"
        "Classification: "
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage:
# Assuming df['prompt_text'] is populated from the previous step.
for prompt in df['prompt_text']:
    result = classify_stress_with_gemini(prompt)
    print(f"Prompt:\n{prompt}\n\nGemini's response:\n{result}\n---\n")


def create_stress_narrative(row):
    """
    Creates a descriptive text from a row of physiological and motion features.
    
    Args:
        row (pd.Series): A row from the feature-engineered DataFrame.
    
    Returns:
        str: A narrative describing the physiological state.
    """
    narrative = (
        f"The participant's average heart rate was {row['avg_hr']:.2f} bpm. "
        f"The heart rate variability (HRV) was characterized by an LF/HF ratio of {row['lf_hf_ratio']:.2f}, "
        f"and an RMSSD of {row['rmssd']:.2f}. "
        f"Motion sensors showed a mean acceleration of {row['avg_accel']:.2f} g and a gyroscope reading of {row['avg_gyro']:.2f} deg/s. "
        f"Respiration was at {row['respiration_rate']:.2f} breaths per minute. "
        f"The physiological markers indicate a response pattern consistent with either physical or mental stress. "
        f"Please classify the stress type based on the provided data and justify your answer."
    )
    return narrative

# # Example of a feature-engineered DataFrame (replace with your actual data)
# data = {'avg_hr': [110, 85, 130],
#         'lf_hf_ratio': [2.5, 0.8, 3.1],
#         'rmssd': [15.2, 55.6, 12.1],
#         'avg_accel': [0.1, 0.9, 0.08],
#         'avg_gyro': [0.05, 1.2, 0.04]}
# df = pd.DataFrame(data)
df = pd.read_csv("")
# Apply the function to your data to create a new 'text' column
df['prompt_text'] = df.apply(create_stress_narrative, axis=1)

print(df['prompt_text'].iloc[0])