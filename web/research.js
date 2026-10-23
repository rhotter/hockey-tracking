(() => {
  'use strict';
  const selector = document.getElementById('ball-example');
  const video = document.getElementById('ball-video');
  SportScansMedia.load(video,video.dataset.mediaSrc);
  const note = document.getElementById('ball-example-note');
  const reviewed = {'buf-1200': 6, 'ns-0600': 3, 'md-0900': 6};
  selector.addEventListener('change', () => {
    const id = selector.value;
    if (!Object.hasOwn(reviewed, id)) return;
    video.pause();
    video.poster = `media/ball/${id}.jpg`;
    SportScansMedia.load(video,`media/ball/${id}.mp4`);
    note.textContent = `Cyan: model predictions. White: reviewed ball locations on ${reviewed[id]} sampled frames. Other frames are unscored.${id === 'md-0900' ? ' This clip includes false detections near floor markings and image edges.' : ''}`;
  });
  video.addEventListener('error', () => {
    note.textContent = 'The evidence clip could not load. Reload the page to try again.';
  });
})();
