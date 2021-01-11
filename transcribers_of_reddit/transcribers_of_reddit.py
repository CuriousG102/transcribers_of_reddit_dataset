"""transcribers_of_reddit dataset."""

import collections
import enum
import os
import re
import urllib

import pandas as pd
import tensorflow_datasets as tfds

# TODO(transcribers_of_reddit): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Screenshots and transcriptions from r/transcribersofreddit volunteers. Process of building this
dataset documented in blog post and this CoLab.
"""

# TODO(transcribers_of_reddit): BibTeX citation
_CITATION = """
"""


class TranscriptionCategory(enum.Enum):
  ART_AND_IMAGES_WITHOUT_TEXT = 1
  IMAGES_WITH_TEXT = 2
  GREENTEXT_4CHAN = 3
  REDDIT_POST = 4
  REDDIT_COMMENT = 5
  FACEBOOK_POST = 6
  FACEBOOK_COMMENT = 7
  TEXT_MESSAGES = 8
  TWITTER_POST = 9
  TWITTER_REPLY = 10
  COMIC = 11
  GIF = 12
  CODE = 13
  MEME = 14
  OTHER = 15

  @classmethod
  def get_category(cls, transcription):
    if '*Image Transcription: Greentext*' in transcription or '*Image Transcription: 4chan*' in transcription:
      return cls.GREENTEXT_4CHAN
    if '*Image Transcription: Reddit*' in transcription:
      return cls.REDDIT_POST
    if '*Image Transcription: Reddit Comments*' in transcription:
      return cls.REDDIT_COMMENT
    if '*Image Transcription: Facebook Post*' in transcription:
      return cls.FACEBOOK_POST
    if '*Image Transcription: Facebook Comments*' in transcription or '*Image Transcription: Facebook Comment*'in transcription:
      return cls.FACEBOOK_COMMENT
    if '*Image Transcription: Text Messages*' in transcription:
      return cls.TEXT_MESSAGES
    if '*Image Transcription: Twitter Post*' in transcription:
      return cls.TWITTER_POST
    if '*Image Transcription: Twitter Post and Replies*' in transcription:
      return cls.TWITTER_REPLY
    if '*Image Transcription: Comic*' in transcription:
      return cls.COMIC
    if '*Image Transcription: GIF*' in transcription:
      return cls.GIF
    if '*Image Transcription: Code*' in transcription:
      return cls.CODE
    if '*Image Transcription: Meme*' in transcription:
      return cls.MEME
    if '*Image Transcription:' in transcription:
      # No text
      if re.search(r'---\s*\[\*.+\*\]\s*---', transcription, re.M):
        return cls.ART_AND_IMAGES_WITHOUT_TEXT
      else:
        return cls.IMAGES_WITH_TEXT

    return cls.OTHER


class TranscribersOfReddit(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for transcribers_of_reddit dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Create files and scrape using CoLab from the description.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(transcribers_of_reddit): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'transcription_category': tfds.features.ClassLabel(names=[
                str(e).replace('TranscriptionCategory.', '') for e in TranscriptionCategory]),
            'transcription': tfds.features.Text(),
        }),
        supervised_keys=None,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(transcribers_of_reddit): Downloads the data and defines the splits
    images_path = dl_manager.extract(os.path.join(
      dl_manager.manual_dir, 'tor_images.zip'))
    comments_path = os.path.join(dl_manager.manual_dir, 'tor_comments.csv')
    submissions_path = os.path.join(dl_manager.manual_dir, 'tor_submissions.csv')

    # TODO(transcribers_of_reddit): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(
          images_path / 'images', comments_path, submissions_path),
    }

  def _generate_examples(self, images_path, comments_path, submissions_path):
    """Yields examples."""
    comments = pd.read_csv(comments_path, low_memory=False).drop(['Unnamed: 0'], axis=1)
    submissions = pd.read_csv(submissions_path, low_memory=False).drop(['Unnamed: 0'],axis=1)
    USABLE_EXTENSIONS = {'jpg', 'png'}
    downloadable_submissions = submissions[
      submissions['url'].apply(
        lambda url: urllib.parse.urlparse(url).path.split('.')[-1] in USABLE_EXTENSIONS)
    ]
    len(downloadable_submissions)
    duplicates = set(
      k for k, v in
      collections.Counter(downloadable_submissions['url'].apply(
        lambda url: urllib.parse.urlparse(url).path.split('/')[-1])).items()
      if v > 1)
    image_files = set(os.listdir(images_path))

    def keep_row(url):
      file = urllib.parse.urlparse(url).path.split('/')[-1]
      return file not in duplicates and file in image_files

    # Get rid of pesky t._ prefeix..
    comments['polite_link_id'] = comments['link_id'].apply(lambda id: id[id.find('_') + 1:])
    full_transcriptions = pd.merge(downloadable_submissions[downloadable_submissions['url'].apply(
      keep_row)], comments.drop_duplicates('polite_link_id'),
      left_on='id', right_on='polite_link_id', suffixes=('_submission', '_comment'))
    full_transcriptions['category'] = full_transcriptions['body'].apply(TranscriptionCategory.get_category)

    # TODO(transcribers_of_reddit): Yields (key, example) tuples from the dataset
    for _, row in full_transcriptions.iterrows():
      image_path = urllib.parse.urlparse(row['url']).path.split('/')[-1]
      yield image_path, {
          'image': images_path / image_path,
          'transcription_category': str(row['category']).replace('TranscriptionCategory.', ''),
          'transcription': row['body']
      }

