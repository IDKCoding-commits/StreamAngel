import requests
#import nemo.collections.asr as nemo_asr
#asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")

#transcriptions = asr_model.transcribe(["file.wav"])

def get_search_results(search_input: str):
    search_input = search_input.replace(" ", "+")
    search = f"https://itunes.apple.com/search?term={search_input}&media=podcast&limit=5"
    response = requests.get(search)
    for item in response.json()['results']:
        artist_name = item['artistName']
        print(artist_name)
        podcast_name = item['collectionName']
        print(podcast_name)
    print(artist_name)
