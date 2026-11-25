import pytest
from backend.main import video_id
@pytest.mark.parametrize('url',['https://www.youtube.com/watch?v=oxxucX4dr5k','https://youtu.be/oxxucX4dr5k'])
def test_valid(url): assert video_id(url)=='oxxucX4dr5k'
@pytest.mark.parametrize('url',['http://youtu.be/oxxucX4dr5k','https://evil.com/watch?v=oxxucX4dr5k','https://youtu.be/short'])
def test_invalid(url):
    with pytest.raises(ValueError): video_id(url)
