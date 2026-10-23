(() => {
 'use strict';
 const clip=window.SportScansTrackingCatalog['pittsburgh-stress-0900'];
 for(const [id,label]of [['vlm-osnet-baseline','VLM comparison · existing appearance links'],['vlm-embedding-2b-joint','VLM comparison · Qwen3-VL Embedding 2B joint images']]){
  if(!clip.methods.includes(id))clip.methods.push(id);
  clip.methodLabels[id]=label;
 }
})();
