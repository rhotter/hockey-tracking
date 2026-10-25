(() => {
 'use strict';
 const methods={"roboflow-rfdetr_nbhl-rf_botsort-default": "RF-DETR + Roboflow BoT-SORT", "roboflow-rfdetr_nbhl-sports_bytetrack-default": "RF-DETR + Sports ByteTrack", "roboflow-yolo_cached-rf_botsort-default": "YOLO + Roboflow BoT-SORT", "roboflow-yolo_cached-sports_bytetrack-default": "YOLO + Sports ByteTrack", "roboflow-yolo_cached-rf_botsort-default-appearance": "YOLO + Roboflow BoT-SORT + existing OSNet links"};
 for(const id of ["pittsburgh-0600", "pittsburgh-1800", "marylandwomen-0600", "marylandwomen-1800", "pittsburgh-stress-0900"]){
 const clip=window.SportScansTrackingCatalog[id];
 for(const [id,label] of Object.entries(methods)){if(!clip.methods.includes(id))clip.methods.push(id);clip.methodLabels[id]=label;clip.methodDescriptions??={};clip.methodDescriptions[id]='Roboflow experiment on NBHL ball hockey. Current detector boxes with automatic fragment IDs; no jersey identity or off-camera interpolation. See the Roboflow replacement report for measured results and limitations.';}
 }
})();
