# patch.py - Fixed version with proper imports

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Import from the location where modeling_minicpmo.py is placed
import sys
sys.path.append("/opt")

from modeling_minicpmo import gen_logits, VoiceChecker, NumberToTextConverter, sentence_end, OmniOutput

# Dummy base class since we're just using the method
class MiniCPMOPatched:
    """
    A patched version of MiniCPMO with optimizations for inference.
    and sets a smaller chunk size for faster inference.
    """

    # def compile_model(self, mode="default", backend=None):
    #     print(f"🔧 Applying optimizations (mode='{mode}')")
        
    #     print("✅ torch.autocast(bfloat16): Applied")  
    #     print("✅ Smaller chunk_size (8): Applied")
    #     print("⚠️ Skipping torch.compile to avoid breaking inference")


    def _generate_mel_spec_audio_streaming(
        self,
        spk_bounds,
        streamer,
        output_chunk_size=25,
        spk_embeds=None,
        prev_seg_text_ids=None,
        prev_seg_text_left="",
        prev_seg_audio_ids=None,
        enable_regenerate=False,
    ):
        # get spk_embedding
        gen_text = ""
        tts_text = ""
        new_segment_gen = False
        if spk_embeds is None:
            spk_bound = spk_bounds[0][-1]
            r = next(streamer)
            txt = r["text"]
            gen_text += txt.split("<|tts_eos|>")[0]
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
            last_hidden_states = r["hidden_states"][0][-1][0]  # output: (input_seq_len, dim)
            spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]

        # init past_key_values
        logits_warpers, logits_processors = gen_logits(
            num_code=626, top_P=self.tts.top_p, top_K=self.tts.top_k, repetition_penalty=self.tts.repetition_penalty
        )
        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )
        tts_start_token_len = 1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs
        dtype = self.tts.emb_text.weight.dtype
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]
        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        # prefill prev segment for smooth
        chunk_idx = 0
        new_ids_len = 0
        prev_text_len = 0
        if prev_seg_text_ids is not None and prev_seg_audio_ids is not None:
            tts_token_lens = prev_seg_text_ids.shape[1]
            # assert tts_token_lens % self.tts.streaming_text_chunk_size == 0
            streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
            position_ids = torch.arange(
                0, tts_token_lens + tts_start_token_len, dtype=torch.long, device=self.tts.device
            ).unsqueeze(0)

            text_input_ids = self.get_tts_text_start_token_ids()
            text_input_ids = torch.cat([text_input_ids, prev_seg_text_ids], dim=1)
            past_key_values = self.tts.prefill_text(
                input_ids=text_input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                lm_spk_emb_last_hidden_states=spk_embeds,
            )
            past_key_values = self.tts.prefill_audio_ids(
                input_ids=prev_seg_audio_ids[:, :-1, :],
                # not prefill last id, which will be input_id of next generation
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
            )

            # update init
            chunk_idx += int(tts_token_lens / self.tts.streaming_text_chunk_size)
            audio_input_ids = torch.cat([audio_input_ids, prev_seg_audio_ids], dim=1)
            text = self.tts_processor.text_tokenizer.decode(prev_seg_text_ids[0].tolist(), add_special_tokens=False)

            gen_text += text
            gen_text += prev_seg_text_left
            prev_text_len = len(gen_text)  # takecare the position
            new_ids_len += prev_seg_audio_ids.shape[1]

        prev_wav = None
        eos_lab = False
        stop = False
        shift_len = 180
        voice_checker = VoiceChecker()
        number_converter = NumberToTextConverter()
        lang = None
        gen_text_raw = gen_text
        for t, r in enumerate(streamer):
            t += 1
            txt = r["text"]
            txt = txt.split("<|tts_eos|>")[0]
            gen_text_raw += txt
            if t == 1 and txt == "" and prev_seg_text_ids is not None:
                logger.warning("New segment is empty, generation finished.")
                return
            if t <= 2:  # do just one time, more token greater certainty
                lang = number_converter.detect_language(gen_text_raw)
            gen_text += number_converter.replace_numbers_with_text(txt, lang).replace("*", "")  # markdown **

            # TODO speed up
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)

            if tts_token_lens >= self.tts.streaming_text_reserved_len - shift_len:
                end_c = sentence_end(txt)
                if end_c:
                    end_c_idx = gen_text.rfind(end_c)
                    assert end_c_idx != -1
                    text_left = gen_text[end_c_idx + 1 :]
                    gen_text = gen_text[: end_c_idx + 1]
                    tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
                    new_segment_gen = True
                    logger.debug(
                        f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, starting a new segment generation"
                    )
                    break

            if tts_token_lens >= (chunk_idx + 1) * self.tts.streaming_text_chunk_size:

                # do prefill and generate
                if chunk_idx == 0:
                    begin = 0
                    end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len
                else:
                    begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
                    end = min(
                        (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len, condition_length - 1
                    )

                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = (
                    outputs.audio_input_ids
                )  # [1,seq_len,4] seq_len=tts.streaming_text_reserved_len + 3 + len(new_ids)
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)  # [1,100,50] -> [50*256]

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = []
                    for i in range(len(past_key_values)):
                        past_kv.append(
                            (
                                past_key_values[i][0][:, :, :-N, :],  # .clone(),
                                past_key_values[i][1][:, :, :-N, :],  # .clone(),
                            )
                        )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=self.force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)

                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)

                    else:
                        prev_wav = wav_np[-512 * 4:]
                        wav_np = wav_np[:-512 * 4]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)
                if outputs.finished:
                    logger.debug("Generation finished.")
                    eos_lab = True
                    break

        if not eos_lab and tts_text:
            logger.debug("eos_lab False, Generation continue.")

            if chunk_idx == 0:
                begin = 0
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
            end = tts_token_lens + tts_start_token_len + 1  # 1 for [Etts]
            if end > begin:
                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )

            while True:
                # temp = [0.1, 0.3, 0.1, 0.3] if chunk_idx < 21 else [0.1] * self.tts.num_vq
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    # temperature=torch.tensor([0.1] * self.tts.num_vq, dtype=torch.float, device=self.tts.device),
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = []
                    for i in range(len(past_key_values)):
                        past_kv.append(
                            (
                                past_key_values[i][0][:, :, :-N, :],  # .clone(),
                                past_key_values[i][1][:, :, :-N, :],  # .clone(),
                            )
                        )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=self.force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)
                    else:

                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)
                    else:
                        prev_wav = wav_np[-512 * 4:]
                        wav_np = wav_np[:-512 * 4]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > 2048:
                    stop = True
                    logger.debug("Generation length > 2048, stopped.")
                    break

        if prev_wav is not None:
            cur_text = gen_text_raw[prev_text_len:]
            yield OmniOutput(text=cur_text, audio_wav=prev_wav, sampling_rate=sr)  # yield last chunk wav without smooth

        if new_segment_gen and not stop:
            logger.debug(
                f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, start a new segment generation"
            )
            tid_len = 5  # self.tts.streaming_text_chunk_size
            prev_seg_text_ids = tts_input_ids[:, end - 1 - tid_len : end - 1]  # exclude last Etts
            aid_len = 50  # int(tid_len * new_ids_len / tts_token_lens)
            prev_seg_audio_ids = outputs.new_ids[:, -aid_len:, :]

            result = self._generate_mel_spec_audio_streaming(
                spk_bounds,
                streamer,
                output_chunk_size,
                spk_embeds,
                prev_seg_text_ids,
                text_left,
                prev_seg_audio_ids,
                enable_regenerate=enable_regenerate,
            )
            for res in result:
                yield res
