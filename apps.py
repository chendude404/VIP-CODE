import torch

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response

# --- Helper Function ---
def generate_zsc_output(prompt):
    classes_verbalized = settings.ZERO_SHOT_LABELS
    zeroshot_classifier = settings.ZERO_SHOT_PIPELINE

    if not classes_verbalized or not zeroshot_classifier:
        return None

    try:
        output = zeroshot_classifier(prompt, classes_verbalized, multi_label=True)
        return output
    except Exception as e:
        print(f"Error during zero-shot classification: {e}") # Keeping essential error prints
        return None

# --- API View ---
@api_view(['POST'])
def classify_text(request):
    text = request.data.get('text', '').strip()
    song_list = request.data.get('song_list', [])

    if not text:
        return Response({"error": "No text provided"}, status=400)

    try:
        # Assuming 'generate_subplaylist' is defined/imported elsewhere
        # Example: from .utils import generate_subplaylist
        print('Calling generate_subplaylist...') # Checkpoint retained
        result = generate_subplaylist(text, song_list)
        print(f'Result is {result}') # Optional: Result check

        return Response({
            'tracks': result
        })

    except NameError as e:
         error_msg = f"Internal configuration error: Required function 'generate_subplaylist' not found. {e}"
         print(f"Error: {error_msg}") # Keep configuration error print
         return Response({"error": error_msg}, status=500)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        print(f"Error: {error_msg}") # Keep generic error print
        return Response({"error": error_msg}, status=500)
