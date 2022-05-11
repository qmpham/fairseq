import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
#from examples.CT2_quant_noise import quantization_utils
from . import quantization_utils
from omegaconf import DictConfig
from fairseq.tasks.translation import TranslationConfig
def quantize_model_scalar(model, model_cfg: DictConfig):
    quant_noise_scalar = getattr(model_cfg, "quant_noise_scalar", 0) or 0
    if quant_noise_scalar > 0:
        # quantize_model edits the model in place
        quantization_utils.quantize_model_(model, p=quant_noise_scalar, bits=8, update_step=1000)
    return model

@register_task("ct2_quant_noise_robustness",dataclass=TranslationConfig)
class CT2_quant_noise_task(TranslationTask):
    def __init__(self, cfg, src_dict, tgt_dict):
        super(CT2_quant_noise_task,self).__init__(cfg, src_dict, tgt_dict)
    def build_model(self, cfg, from_checkpoint=False):
        from fairseq import models        
        print("hellllllloooooooooooo")
        model = models.build_model(cfg, self, from_checkpoint)
        model = quantize_model_scalar(model, cfg)

        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model


    
    
    