from aylienapiclient import textapi


class AngstClassifier:

    def __init__(self, app_id, app_key):
        self.client = textapi.Client(app_id, app_key)

    def analyze_text(self, text):
        sentiment = self.client.Sentiment({'text': text.replace('\n', ' ')})
        return sentiment



if __name__=='__main__':
    text = """Nobody heard him, the dead man,
But still he lay moaning:
I was much further out than you thought
And not waving but drowning.

Poor chap, he always loved larking
And now he’s dead
It must have been too cold for him his heart gave way,
They said.

Oh, no no no, it was too cold always
(Still the dead one lay moaning)
I was much too far out all my life
And not waving but drowning."""
    angst = AngstClassifier('8d1fc860', 'e87b5298692123977e5c6cc98c6dae0e')
    print(angst.analyze_text(text))


