#!/usr/bin/env python3
"""CPU regression checks for Add, adaptive Add and feedback ablations."""
import unittest
import torch
from hmnet.models.efficientvit_tasks import build_frame_task
from hmnet.models.base.backbone.efficientvit_b1 import EfficientViTB1
from hmnet.models.base.backbone.adaptive_add import AdaptiveAdd


class FusionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_model_initialization_matches_add(self):
        torch.manual_seed(42)
        add = build_frame_task("segmentation", fusion_mode="add").eval()
        rng_add = torch.get_rng_state().clone()
        torch.manual_seed(42)
        gated = build_frame_task("segmentation", fusion_mode="adaptive_add").eval()
        self.assertTrue(torch.equal(rng_add, torch.get_rng_state()))
        a, g = add.state_dict(), gated.state_dict()
        self.assertTrue(all(torch.equal(t, g[k]) for k, t in a.items()))
        extra = set(g) - set(a)
        self.assertEqual(len(extra), 8)
        self.assertTrue(all(torch.count_nonzero(g[k]) == 0 for k in extra))
        event, rgb = torch.randn(1,20,64,96), torch.randn(1,3,64,96)
        with torch.no_grad():
            fa, fg = add.backbone(event,rgb), gated.backbone(event,rgb)
            for x,y in zip(fa,fg):self.assertTrue(torch.equal(x,y))
            meta=[dict(height=64,width=96)]
            ya=add.seg_head(add.neck(list(fa)),meta)
            yg=gated.seg_head(gated.neck(list(fg)),meta)
            self.assertTrue(torch.equal(ya,yg))

    def test_gate_gradients_and_extremes(self):
        gate=AdaptiveAdd()
        e=torch.randn(2,8,9,11,requires_grad=True)
        r=torch.randn_like(e,requires_grad=True)
        self.assertTrue(torch.equal(gate(e,r),e+r))
        gate(e,r).square().mean().backward()
        for p in (gate.weight,gate.bias,e,r):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)
        with torch.no_grad():
            gate.bias.fill_(100)
            self.assertTrue(torch.equal(gate(e,r),2*r))
            gate.bias.fill_(-100)
            self.assertTrue(torch.equal(gate(e,r),2*e))

    def test_feedback_switch_and_modality_isolation(self):
        e,r=torch.randn(1,20,64,96),torch.randn(1,3,64,96)
        for mode,feedback in [("cross_stage_post_mbconv",True),
                              ("cross_stage_post_mbconv_no_feedback",False)]:
            torch.manual_seed(42)
            model=EfficientViTB1(fusion=True,fusion_mode=mode).eval()
            seen={}
            def save(name):
                def hook(module,inputs,output):seen[name]=output.detach().clone()
                return hook
            def save_input(module,inputs):seen['stage2_input']=inputs[0].detach().clone()
            def interaction_hook(module,inputs,output):seen['rgb_next']=output[0].detach().clone()
            hs=[model.rgb_encoder.stages[0].register_forward_hook(save('stage1')),
                model.rgb_encoder.stages[1].register_forward_pre_hook(save_input),
                model.interactions[0].register_forward_hook(interaction_hook),
                model.rgb_encoder.stages[3].register_forward_hook(save('rgb4'))]
            with torch.no_grad():first=model(e,r)
            expected=seen['rgb_next'] if feedback else seen['stage1']
            self.assertTrue(torch.equal(seen['stage2_input'],expected))
            if not feedback:
                last=seen['rgb4'].clone()
                with torch.no_grad():second=model(e+0.3,r)
                self.assertTrue(torch.equal(last,seen['rgb4']))
                self.assertFalse(torch.equal(first[3],second[3]))
                with torch.no_grad():
                    ev,im=model.event_encoder(e),model.rgb_encoder(r)
                    reference=tuple(model.relu(proj(inter(im[f"stage{i+1}"],ev[f"stage{i+1}"])[2]))
                                    for i,(inter,proj) in enumerate(zip(model.interactions,model.fused_proj)))
                for x,y in zip(first,reference):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for h in hs:h.remove()

    def test_checkpoint_recomputation_bn_once(self):
        from hmnet.models.base.backbone.cross_modal_litemla import CrossModalLiteMLA
        layer=CrossModalLiteMLA(32).train()
        e=torch.randn(2,32,8,8,requires_grad=True);r=torch.randn_like(e,requires_grad=True)
        layer(r,e)[2].square().mean().backward()
        for module in layer.modules():
            if isinstance(module,torch.nn.BatchNorm2d):self.assertEqual(module.num_batches_tracked.item(),1)
        self.assertGreater(e.grad.abs().sum().item(),0)
        self.assertGreater(r.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))

if __name__=="__main__":unittest.main(verbosity=2)
