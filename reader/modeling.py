import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import ElectraModel, ElectraPreTrainedModel, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


# Bi-LSTM
class ElectraBiLSTMForQuestionAnswering(ElectraPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.electra = ElectraModel(config)
		self.lstm = nn.LSTM(input_size=config.hidden_size,
		                    hidden_size=config.hidden_size,
		                    num_layers=2,
		                    batch_first=False,
		                    bidirectional=True)
		self.qa_outputs = nn.Linear(config.hidden_size*2, config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		start_positions=None,
		end_positions=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Bert model returns
		outputs = self.electra(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		# Bert model last_hidden_state
		# [batch_size, sequence_length, hidden_size]
		sequence_output = outputs[0]

		# Bi-LSTM output
		# hidden shape = [batch_size, sequence_length, hidden_size * 2]
		hidden, (last_hidden, last_cell) = self.lstm(sequence_output)

		# Linear output
		# logits shape = [batch_size, sequence_length, 2]
		logits = self.qa_outputs(hidden)

		# start_logits shape = [batch_size, sequence_length]
		# end_logits shape = [batch_size, sequence_length]
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1).contiguous()
		end_logits = end_logits.squeeze(-1).contiguous()

		# loss
		total_loss = None
		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions = start_positions.clamp(0, ignored_index)
			end_positions = end_positions.clamp(0, ignored_index)

			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2

		if not return_dict:
			output = (start_logits, end_logits) + outputs[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return QuestionAnsweringModelOutput(
			loss=total_loss,
			start_logits=start_logits,
			end_logits=end_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


# Bi-LSTM
class ElectraBiGRUForQuestionAnswering(ElectraPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.electra = ElectraModel(config)
		classifier_dropout = config.hidden_dropout_prob
		self.dropout = nn.Dropout(classifier_dropout)
		self.bi_gru = nn.GRU(config.hidden_size,
		                     config.hidden_size,
		                     num_layers=2,
		                     dropout=classifier_dropout,
		                     batch_first=True,
		                     bidirectional=True)
		self.qa_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		start_positions=None,
		end_positions=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Bert model returns
		outputs = self.electra(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]
		sequence_output, hc = self.bi_gru(sequence_output)
		logits = self.qa_outputs(sequence_output)

		# start_logits shape = [batch_size, sequence_length]
		# end_logits shape = [batch_size, sequence_length]
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1).contiguous()
		end_logits = end_logits.squeeze(-1).contiguous()

		# loss
		total_loss = None
		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions = start_positions.clamp(0, ignored_index)
			end_positions = end_positions.clamp(0, ignored_index)

			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2

		if not return_dict:
			output = (start_logits, end_logits) + outputs[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return QuestionAnsweringModelOutput(
			loss=total_loss,
			start_logits=start_logits,
			end_logits=end_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)



class RobertaBiGRUForQuestionAnswering(RobertaPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.roberta = RobertaModel(config)
		classifier_dropout = config.hidden_dropout_prob
		self.dropout = nn.Dropout(classifier_dropout)
		self.bi_gru = nn.GRU(config.hidden_size,
		                     config.hidden_size,
		                     num_layers=2,
		                     dropout=classifier_dropout,
		                     batch_first=True,
		                     bidirectional=True)
		self.qa_outputs = nn.Linear(config.hidden_size * 2, config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		start_positions=None,
		end_positions=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Bert model returns
		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]
		sequence_output, hc = self.bi_gru(sequence_output)
		logits = self.qa_outputs(sequence_output)

		# start_logits shape = [batch_size, sequence_length]
		# end_logits shape = [batch_size, sequence_length]
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1).contiguous()
		end_logits = end_logits.squeeze(-1).contiguous()

		# loss
		total_loss = None
		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions = start_positions.clamp(0, ignored_index)
			end_positions = end_positions.clamp(0, ignored_index)

			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2

		if not return_dict:
			output = (start_logits, end_logits) + outputs[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return QuestionAnsweringModelOutput(
			loss=total_loss,
			start_logits=start_logits,
			end_logits=end_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
