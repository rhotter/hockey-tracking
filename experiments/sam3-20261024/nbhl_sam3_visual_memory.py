"""Experimental reuse of initial exemplar geometry tokens across video frames."""


def install(detector):
    import hashlib
    import torch
    original = detector._encode_prompt
    memory = {}
    stats = {'calls':0,'encodings':0,'policy':'Initial exemplar geometry tokens retained; no moving box prompt or weight update.'}

    def encode(backbone_out,find_input,geometric_prompt,**kwargs):
        assert kwargs.get('visual_prompt_embed') is None
        if not memory:
            assert geometric_prompt.box_embeddings is not None and geometric_prompt.box_embeddings.numel()>0
            backbone_out,features,positions,sizes = detector._get_img_feats(backbone_out,find_input.img_ids)
            tokens,mask = detector.geometry_encoder(geo_prompt=geometric_prompt,img_feats=features,
                                                    img_sizes=sizes,img_pos_embeds=positions)
            assert torch.isfinite(tokens).all()
            memory.update(tokens=tokens.detach().clone(),mask=mask.detach().clone())
            stats.update(encodings=1,shape=list(tokens.shape),
                         token_sha256=hashlib.sha256(tokens.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        stats['calls']+=1
        kwargs.update(visual_prompt_embed=memory['tokens'],visual_prompt_mask=memory['mask'])
        return original(backbone_out,find_input,detector._get_dummy_prompt(),**kwargs)

    detector._encode_prompt=encode
    return stats
