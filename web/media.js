(() => {
 'use strict';
 const states=new WeakMap();
 function clear(video){
  const previous=states.get(video);if(previous){previous.controller.abort();previous.status.remove();}
  video.pause();video.removeAttribute('src');video.load();
  if(previous?.objectURL)URL.revokeObjectURL(previous.objectURL);
  states.delete(video);delete video.dataset.mediaReady;
 }
 function load(video,url,{autoplay=false}={}){
  const previous=states.get(video);
  if(previous?.url===url&&!previous.failed){
   if(autoplay)previous.promise.then(ok=>{if(ok)video.play().catch(()=>{});});
   return previous.promise;
  }
  clear(video);video.dataset.mediaSource=url;
  const status=document.createElement('p');status.className='note media-loading';status.setAttribute('role','status');status.textContent='Loading clip…';const anchor=video.parentElement.classList.contains('replay-video')?video.parentElement:video;anchor.insertAdjacentElement('afterend',status);
  const state={url,status,controller:new AbortController(),objectURL:null,failed:false};states.set(video,state);
  state.promise=(async()=>{
   try{
    const response=await fetch(url,{signal:state.controller.signal});if(!response.ok||response.headers.get('content-type')?.startsWith('text/'))throw Error('Clip unavailable');
    const blob=await response.blob();if(!blob.size)throw Error('Clip empty');
    if(states.get(video)!==state)return false;
    state.objectURL=URL.createObjectURL(blob);video.src=state.objectURL;video.load();video.dataset.mediaReady=url;status.remove();
    if(autoplay)await video.play().catch(()=>{});
    return true;
   }catch(error){
    if(error.name!=='AbortError'&&states.get(video)===state){state.failed=true;status.textContent='This clip could not load. Try selecting it again or reload the page.';status.setAttribute('role','alert');}
    return false;
   }
  })();return state.promise;
 }
 window.SportScansMedia={load,clear};
})();
