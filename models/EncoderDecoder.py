import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning import LightningModule
from utils.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler


class EncoderDecoder(LightningModule):
    """
    Encoder Decoder
    """

    def __init__(self, config, tokenizer, transformer, dataset_reader):
        """
        :param config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = transformer
        self.dataset_reader = dataset_reader

        self.use_deepspeed = self.config.compute_strategy.startswith("deepspeed")
        self.use_ddp = self.config.compute_strategy.startswith("ddp")
        self.load_model()

        self._last_global_step_saved = -1

    def training_step(self, batch, batch_idx):

        if self.config.mc_loss > 0 or self.config.unlikely_loss > 0:

            verification_input_ids, choices_ids, v_labels = batch["verification_input_ids"], batch["answer_choices_ids"], batch["v_labels"]
            idxs = batch["idx"]

            num_choices = self.config.n_ways

            if self.config.stage==2:
                with torch.no_grad():
                    pred_output = self.predict(batch,if_inner=True)[0]
                    claim_choice_score = pred_output['choices_scores']
                    claim_idx = pred_output['idx'].tolist()
                    claim_choice_score = torch.tensor(claim_choice_score).to(verification_input_ids[0].device)
                    claim_choice_score = claim_choice_score.unsqueeze(0)
                    claim_pred = pred_output['prediction']
                    con_l = {0:[0,0,2,2],1:[1,1,1,0],2:[2,2,0,2]}
                    pre_labels = {idxs[x].item():claim_pred[xi] for xi,x in enumerate(claim_idx) }
                    
                    for kk in claim_idx:
                        k = idxs[kk].item()
                        if not self.config.zero_shot:
                            pre_labels[k] = v_labels[kk].item()
                        pre_labels[k+1] = con_l[pre_labels[k]][1]
                        pre_labels[k+2] = con_l[pre_labels[k]][2]
                        pre_labels[k+3] = con_l[pre_labels[k]][3]
                    
                    for ii,iidx in enumerate(idxs):
                        v_labels[ii]=pre_labels[iidx.cpu().item()]

                if self.config.zero_shot:    
                    select_idx = (idxs%10!=0).nonzero().squeeze(1)
                    verification_input_ids = torch.index_select(verification_input_ids,0,select_idx)
                    choices_ids = torch.index_select(choices_ids,0,select_idx)
                    v_labels = torch.index_select(v_labels,0,select_idx)

            input_ids_ = verification_input_ids
            labels_ = v_labels
            
            
            def get_encode_states(input_ids_1,nc=self.config.n_ways):                 
                attention_mask = (input_ids_1 != self.tokenizer.pad_token_id).float()
                encoder_hidden_states_ = self.model.encoder(input_ids=input_ids_1, attention_mask=attention_mask)
                encoder_hidden_states1 = encoder_hidden_states_[0]
                encoder_hidden_states = encoder_hidden_states1.unsqueeze(dim=1).repeat(1, nc, 1, 1).flatten(0, 1)
                attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, nc, 1).flatten(0, 1)
                return attention_mask,encoder_hidden_states
            
            def get_decode_outputs(attention_mask_,flat_ids_,encoder_hidden_states_):
                decoder_input_ids = torch.cat([torch.zeros_like(flat_ids_[:, :1]), flat_ids_[:, :-1]], dim=1)
                decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
                
                model_output = self.model(
                    attention_mask=attention_mask_,
                    encoder_outputs=[encoder_hidden_states_],
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                return model_output
            
            attention_masks,encoder_hidden_states = get_encode_states(input_ids_,num_choices)

            flat_choices_ids,lm_targets,lm_target1s,flat_ids = [],[],[],[]
            choices_ids = [choices_ids] if not isinstance(choices_ids,list) else choices_ids
            for choices_ids_sub in choices_ids:
                flat_choices_ids_ = choices_ids_sub.flatten(0, 1)
                lm_target_ = flat_choices_ids_ - 100 * (flat_choices_ids_ == self.tokenizer.pad_token_id).long()
                flat_ids_ = flat_choices_ids_
                lm_target1_ = lm_target_
                flat_choices_ids.append(flat_choices_ids_)
                lm_targets.append(lm_target_)
                lm_target1s.append(lm_target1_)
                flat_ids.append(flat_ids_)


            attention_masks = [attention_masks] if not isinstance(attention_masks,list) else attention_masks
            encoder_hidden_states = [encoder_hidden_states] if not isinstance(encoder_hidden_states,list) else encoder_hidden_states
            num_choices = [num_choices] if not isinstance(num_choices,list) else num_choices
            labels_ = [labels_] if not isinstance(labels_,list) else labels_
            model_outputs = []
            loss = 0.0
            for attention_mask,encoder_hidden_state,flat_id,flat_choices_id,lm_target,choices_id,nc,lm_target1,labels_1 in \
                zip(attention_masks,encoder_hidden_states,flat_ids,flat_choices_ids,lm_targets,choices_ids,num_choices,lm_target1s,labels_):

                model_output = get_decode_outputs(attention_mask,flat_id,encoder_hidden_state)
                model_outputs.append(model_output)
                bs1 = labels_1.size()[0] 
                choices_scores = (
                    torch.reshape(F.cross_entropy(model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                    ,(bs1, nc, -1))
                    .sum(dim=-1)
                )
                if self.config.length_norm > 0:
                    choices_scores = choices_scores / torch.pow(
                        (choices_id != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                    )
                tensorboard_logs = {}

                labels_1_t = labels_1.squeeze(1) if len(labels_1.size())==2 else labels_1
                lm_loss = F.cross_entropy(
                    torch.reshape(model_output.logits,(bs1, nc, *model_output.logits.size()[1:]))[range(bs1), labels_1_t].flatten(
                        0, 1
                    ),
                    torch.reshape(lm_target1,(bs1, nc, -1))[range(bs1), labels_1_t].flatten(0, 1),
                )

                tensorboard_logs = {"lm_loss": lm_loss.item()}
                if self.config.mc_loss > 0 and nc==self.config.n_ways:
                    labels_1_t = labels_1.squeeze(1) if len(labels_1.size())==2 else labels_1
                    mc_loss = F.cross_entropy(-choices_scores, labels_1_t)
                    tensorboard_logs["mc_loss"] = mc_loss.item()
                else:
                    mc_loss = 0.0

                if self.config.unlikely_loss > 0:
                    cand_loglikely = -torch.reshape(F.cross_entropy(
                        model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1), lm_target1.flatten(0, 1), reduction="none"
                    ),(bs1, nc, -1))
                    cand_loglikely += torch.reshape((lm_target1 < 0),(bs1, nc, -1)) * -100
                    labels_1_t = labels_1.squeeze(1) if len(labels_1.size())==2 else labels_1
                    cand_loglikely[range(bs1), labels_1_t] = -100

                    unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
                    tensorboard_logs["unlikely_loss"] = unlikely_loss.item()
                else:
                    unlikely_loss = 0.0

                tmp = lm_loss + mc_loss * self.config.mc_loss + unlikely_loss * self.config.unlikely_loss
                loss += tmp
            tensorboard_logs["loss"] = loss.item()

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            self.log_dict(tensorboard_logs)

        if self.global_step % self.config.save_step_interval == 0:
            self.save_model()

        return loss

    def predict(self, batch,if_inner=False):

        verification_input_ids, choices_ids, v_labels = batch["verification_input_ids"], batch["answer_choices_ids"], batch["v_labels"]
        idxs = batch["idx"]
        num_choices = self.config.n_ways

        if if_inner:
            select_idx = (idxs%10==0).nonzero().squeeze(1)
            verification_input_ids = torch.index_select(verification_input_ids,0,select_idx)
            choices_ids = torch.index_select(choices_ids,0,select_idx)
            v_labels = torch.index_select(v_labels,0,select_idx)

        input_ids_ = verification_input_ids
        labels_ = v_labels
        
        def get_encode_states(input_ids_1,nc=self.config.n_ways):
            attention_mask = (input_ids_1 != self.tokenizer.pad_token_id).float()
            encoder_hidden_states_ = self.model.encoder(input_ids=input_ids_1, attention_mask=attention_mask)
            encoder_hidden_states1 = encoder_hidden_states_[0]
            encoder_hidden_states = encoder_hidden_states1.unsqueeze(dim=1).repeat(1, nc, 1, 1).flatten(0, 1)
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, nc, 1).flatten(0, 1)
            return attention_mask,encoder_hidden_states
        
        def get_decode_outputs(attention_mask_,flat_ids_,encoder_hidden_states_):

            decoder_input_ids = torch.cat([torch.zeros_like(flat_ids_[:, :1]), flat_ids_[:, :-1]], dim=1)

            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            
            model_output = self.model(
                attention_mask=attention_mask_,
                encoder_outputs=[encoder_hidden_states_],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            return model_output

        attention_masks,encoder_hidden_states = get_encode_states(input_ids_,num_choices)

        input_idx_lists=[]
        input_idx_lists.append([k for k in idxs])

        flat_choices_ids,lm_targets,lm_target1s,flat_ids = [],[],[],[]

        choices_ids = [choices_ids] if not isinstance(choices_ids,list) else choices_ids
        for choices_ids_sub in choices_ids:
            flat_choices_ids_ = choices_ids_sub.flatten(0, 1)
            lm_target_ = flat_choices_ids_ - 100 * (flat_choices_ids_ == self.tokenizer.pad_token_id).long()
            flat_ids_ = flat_choices_ids_
            lm_target1_ = lm_target_
            flat_choices_ids.append(flat_choices_ids_)
            lm_targets.append(lm_target_)
            lm_target1s.append(lm_target1_)
            flat_ids.append(flat_ids_)

        attention_masks = [attention_masks] if not isinstance(attention_masks,list) else attention_masks
        encoder_hidden_states = [encoder_hidden_states] if not isinstance(encoder_hidden_states,list) else encoder_hidden_states
        num_choices = [num_choices] if not isinstance(num_choices,list) else num_choices
        labels_ = [labels_] if not isinstance(labels_,list) else labels_
        model_outputs = []
        batch_output = []
        for attention_mask,encoder_hidden_state,flat_id,flat_choices_id,lm_target,choices_id,nc,lm_target1,labels_1,idx_sub in \
            zip(attention_masks,encoder_hidden_states,flat_ids,flat_choices_ids,lm_targets,choices_ids,num_choices,lm_target1s,labels_,input_idx_lists):

            model_output = get_decode_outputs(attention_mask,flat_id,encoder_hidden_state)
            model_outputs.append(model_output)

            bs1 = labels_1.size()[0]

            choices_scores = (
                torch.reshape(F.cross_entropy(model_output.logits[:,:flat_choices_id.size()[-1],:].flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                ,(bs1, nc, -1))
                .sum(dim=-1)
            )

            if self.config.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_id != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                )
            choices_scores_o = choices_scores.tolist()
            pred_score, prediction = choices_scores.min(dim=1)
            score_gt = choices_scores[range(bs1), labels_1]
            choices_scores[range(bs1), labels_1] = choices_scores.max(dim=-1)[0]
            score_cand = choices_scores.min(dim=-1)[0]

            batch_output_sub = {
                "prediction": prediction.tolist(),
                "pred_scores": pred_score.tolist(),
                "labels": labels_1.tolist(),
                "idx": [k.item() for k in idx_sub],
                "log.score_gt": score_gt.tolist(),
                "log.score_cand": score_cand.tolist(),
            }
            if if_inner:
                batch_output_sub = {
                "prediction": prediction.tolist(),
                "choices_scores": choices_scores_o,
                "labels": labels_1.tolist(),
                "idx": select_idx,
            }
            batch_output.append(batch_output_sub)
        return batch_output

    def validation_step(self, batch, batch_idx):
        batch_output = self.predict(batch)
        return batch_output

    def validation_epoch_end(self, outputs):
        # exchange outputs between processes
        if self.use_deepspeed or self.use_ddp:
            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, outputs)
            if dist.get_rank() == 0:
                if isinstance(outputs[0],list):
                    tmp = []
                    for batch_output in outputs:
                        for outputs1 in gathered_outputs:
                            tmp+=outputs1
                    outputs = tmp
                else:
                    outputs = [batch_output for outputs in gathered_outputs for batch_output in outputs] #origs

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            # let rank 0 collect all outputs
            # now output should be two types
            if isinstance(outputs[0],list):
                accumulated = {key: [] for key in outputs[0][0].keys()}
                for batch_output1 in outputs:
                    for batch_output in batch_output1:
                        for key, value in batch_output.items():
                            accumulated[key].extend(value)
            else:
                accumulated = {key: [] for key in outputs[0].keys()}
                for batch_output in outputs:
                    for key, value in batch_output.items():
                        accumulated[key].extend(value)


            # multi-process may yield dupliated examples in the last batch
            valid_mask = []
            idx_set = set()
            for ii,idx in enumerate(accumulated["idx"]):
                valid_mask.append(idx not in idx_set)
                idx_set.add(idx)
            for key, values in accumulated.items():
                accumulated[key] = [v for v, m in zip(values, valid_mask) if m]
            
            # compute and log results
            metrics = self.dataset_reader.compute_metric(accumulated)

        else:
            metrics = {}

        self.save_model()
        
        return metrics

    def configure_optimizers(self):
        optimizer, self.trainable_param_names = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_end(self):
        self.save_model(finish=True)

    def load_model(self):
        if self.config.load_weight != "":
            trainable_states = torch.load(self.config.load_weight, map_location=torch.device("cpu"))
            load_result = self.model.load_state_dict(trainable_states, strict=False)
            assert (
                len(load_result.unexpected_keys) == 0
            ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def save_model(self, finish=False):
        if self.config.save_model and (finish or self._last_global_step_saved != self.global_step):
            if finish:
                model_fname = os.path.join(self.config.exp_dir, "finish.pt")
            else:
                model_fname = os.path.join(self.config.exp_dir, f"global_step{self.global_step}.pt")

            if self.use_deepspeed or self.use_ddp:
                torch.distributed.barrier()
                if dist.get_rank() == 0:
                    trainable_states = {
                        param_name: param_weight.cpu()
                        for param_name, param_weight in self.model.state_dict().items()
                        if param_name in self.trainable_param_names
                    }
                    torch.save(trainable_states, model_fname)
                    print(model_fname)
            else:
                trainable_states = {
                    param_name: param_weight.cpu()
                    for param_name, param_weight in self.model.state_dict().items()
                    if param_name in self.trainable_param_names
                }
                torch.save(trainable_states, model_fname)

            self._last_global_step_saved = self.global_step

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        pass
