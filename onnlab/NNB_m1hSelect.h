#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "CustomBackPropgI.h"
#include "SelectableInputI.h"

#include <vector>
#include <algorithm>

namespace nn
{
	template<bool standalone>
	class NNB_m1hSelect : public interfaces::NeuronBasicInterface, public interfaces::CustomBackPropogableInterface, public interfaces::BasicBackPropogableInterface {
		float single_output;
		float backprop_error_accumulator;
		std::vector<interfaces::ConnectionBasicInterface *> outputs;
		std::vector<interfaces::ConnectionBasicInterface *> inputs;

		std::vector<Candidate> output_values;

		std::vector<float> batch_analyzer_input_activate_min_distance;
		unsigned single_output_index;
		bool batch_analyzer_disabled;

		void AddInputConnection(interfaces::ConnectionBasicInterface *input) override {
			inputs.push_back(input);
		}

		void AddOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.push_back(output);
		}

		void RemoveInputConnection(interfaces::ConnectionBasicInterface *input) override {
			inputs.erase(std::remove(inputs.begin(), inputs.end(), input), inputs.end());
		}

		void RemoveOutputConnection(interfaces::ConnectionBasicInterface *output) override {
			outputs.erase(std::remove(outputs.begin(), outputs.end(), output), outputs.end());
		}

		NNB_m1hSelect(const NNB_m1hSelect &) = delete;
		NNB_m1hSelect &operator=(const NNB_m1hSelect &) = delete;
	public:
		const bool is_standalone = standalone;

		NNB_m1hSelect() {
			single_output = 0;
			backprop_error_accumulator = 0;
			batch_analyzer_disabled = true;
			single_output_index = std::numeric_limits<unsigned>::max();
		}

		~NNB_m1hSelect() override {
			for (auto inp : inputs) {
				inp->~ConnectionBasicInterface();
			}
			for (auto out : outputs) {
				out->~ConnectionBasicInterface();
			}
		}

		const std::vector<interfaces::ConnectionBasicInterface *> &InputConnections() override {
			return inputs;
		}
		const std::vector<interfaces::ConnectionBasicInterface *> &OutputConnections() override {
			return outputs;
		}

		bool IsTrainable() override {
			return true;
		}

		float ActivationFunction(float x) const override {
			return x;
		}

		float ActivationFunctionDerivative(float x) const override {
			return 1.0f;
		}

		void UpdateOwnLevel() override {
			output_values.clear();
			float value;
			for (const auto inp : inputs) {
				value = inp->From()->OwnLevel();
				if (!std::isnan(value)) {
					output_values.emplace_back(value, inp);
				}
			}

			if constexpr (standalone) {
				if (output_values.size() == 1) {
					auto &out = output_values[0];
					single_output = out.value;
					if (batch_analyzer_disabled) {
						for (auto item : inputs) {
							if (item != out.id) {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(false); // Set disposed to unused inputs
							} else {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
							}
						}
					} else {
						auto bit = inputs.begin();
						nn::interfaces::SelectableInputInterface *iface;
						float temp_dist;
						unsigned ptrdiff;
						for (auto cit = inputs.begin(), eit = inputs.end(); cit != eit; ++cit) {
							if (*cit != out.id) {
								iface = dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From());
								iface->SetBackPropEnabledState(false); // Set disposed to unused inputs
								
								temp_dist = iface->Accumulator_unNaN_distance();
								ptrdiff = cit - bit;
								if (temp_dist < batch_analyzer_input_activate_min_distance[ptrdiff]) {
									batch_analyzer_input_activate_min_distance[ptrdiff] = temp_dist;
								}
							} else {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
								ptrdiff = cit - bit;
								batch_analyzer_input_activate_min_distance[ptrdiff] = 0.0f;
								single_output_index = ptrdiff;
							}
						}
					}
				} else if (output_values.empty()) {
					nn::interfaces::CBI *lowest_dist_conn = nullptr;
					float lowest_dist = std::numeric_limits<float>::max();
					float dist;
					for (const auto inp : inputs) {
						dist = dynamic_cast<nn::interfaces::SelectableInputInterface *>(inp->From())->Accumulator_unNaN_distance();
						if (dist < lowest_dist) {
							lowest_dist = dist;
							lowest_dist_conn = inp;
						}
					}
					dynamic_cast<nn::interfaces::SelectableInputInterface *>(lowest_dist_conn->From())->Accumulator_make_unNaN();
					lowest_dist_conn->From()->UpdateOwnLevel();
					single_output = lowest_dist_conn->From()->OwnLevel();
					if (batch_analyzer_disabled) {
						for (auto item : inputs) {
							if (item != lowest_dist_conn) {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(false); // Set disposed to unused inputs
							} else {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
							}
						}
					} else {
						auto bit = inputs.begin();
						nn::interfaces::SelectableInputInterface *iface;
						float temp_dist;
						unsigned ptrdiff;
						for (auto cit = inputs.begin(), eit = inputs.end(); cit != eit; ++cit) {
							if (*cit != lowest_dist_conn) {
								iface = dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From());
								iface->SetBackPropEnabledState(false); // Set disposed to unused inputs

								temp_dist = iface->Accumulator_unNaN_distance();
								ptrdiff = cit - bit;
								if (temp_dist < batch_analyzer_input_activate_min_distance[ptrdiff]) {
									batch_analyzer_input_activate_min_distance[ptrdiff] = temp_dist;
								}
							} else {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
								ptrdiff = cit - bit;
								batch_analyzer_input_activate_min_distance[ptrdiff] = 0.0f;
								single_output_index = ptrdiff;
							}
						}
					}
				} else {
					void *largest_dist_conn = nullptr;
					float largest_dist = -1;
					float dist;
					for (const auto &inp : output_values) {
						dist = dynamic_cast<nn::interfaces::SelectableInputInterface *>(reinterpret_cast<nn::interfaces::CBI *>(inp.id)->From())->Accumulator_NaN_distance();
						if (dist > largest_dist) {
							largest_dist = dist;
							largest_dist_conn = inp.id;
						}
					}
					for (const auto &inp : output_values) {
						if (inp.id != largest_dist_conn) {
							dynamic_cast<nn::interfaces::SelectableInputInterface *>(reinterpret_cast<nn::interfaces::CBI *>(inp.id)->From())->Accumulator_make_NaN();
						}
					}
					single_output = reinterpret_cast<nn::interfaces::CBI *>(largest_dist_conn)->From()->OwnLevel();
					if (batch_analyzer_disabled) {
						for (auto item : inputs) {
							if (item != largest_dist_conn) {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(false); // Set disposed to unused inputs
							} else {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
							}
						}
					} else {
						auto bit = inputs.begin();
						nn::interfaces::SelectableInputInterface *iface;
						float temp_dist;
						unsigned ptrdiff;
						for (auto cit = inputs.begin(), eit = inputs.end(); cit != eit; ++cit) {
							if (*cit != largest_dist_conn) {
								iface = dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From());
								iface->SetBackPropEnabledState(false); // Set disposed to unused inputs

								temp_dist = iface->Accumulator_unNaN_distance();
								ptrdiff = cit - bit;
								if (temp_dist < batch_analyzer_input_activate_min_distance[ptrdiff]) {
									batch_analyzer_input_activate_min_distance[ptrdiff] = temp_dist;
								}
							} else {
								dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
								ptrdiff = cit - bit;
								batch_analyzer_input_activate_min_distance[ptrdiff] = 0.0f;
								single_output_index = ptrdiff;
							}
						}
					}
				}
			} else {
				if (output_values.size() == 1) {
					single_output = output_values[0].value;
				} else if (output_values.empty()) {
					nn::interfaces::CBI *lowest_dist_conn = nullptr;
					float lowest_dist = std::numeric_limits<float>::max();
					float dist;
					for (const auto inp : inputs) {
						dist = dynamic_cast<nn::interfaces::SelectableInputInterface *>(inp->From())->Accumulator_unNaN_distance();
						if (dist < lowest_dist) {
							lowest_dist = dist;
							lowest_dist_conn = inp;
						}
					}
					dynamic_cast<nn::interfaces::SelectableInputInterface *>(lowest_dist_conn->From())->Accumulator_make_unNaN();
					lowest_dist_conn->From()->UpdateOwnLevel();
					single_output = lowest_dist_conn->From()->OwnLevel();
					output_values.emplace_back(single_output, lowest_dist_conn);
				} else {
					single_output = std::numeric_limits<float>::quiet_NaN();
				}
				if (!batch_analyzer_disabled) {
					std::vector<unsigned> result_candidates;
					auto bit = inputs.begin();
					float temp_dist;
					unsigned ptrdiff;
					for (auto cit = inputs.begin(), eit = inputs.end(); cit != eit; ++cit) {
						temp_dist = dynamic_cast<nn::interfaces::SelectableInputInterface *>((*cit)->From())->Accumulator_unNaN_distance();
						ptrdiff = cit - bit;
						if (temp_dist < batch_analyzer_input_activate_min_distance[ptrdiff]) {
							batch_analyzer_input_activate_min_distance[ptrdiff] = temp_dist;
						}
						if (temp_dist < 1e-10f) {
							result_candidates.push_back(ptrdiff);
						}
					}
					if (result_candidates.size() == 1) {
						single_output_index = result_candidates[0];
					} else {
						float shortest_dist = std::numeric_limits<float>::max();
						for (auto inp : result_candidates) {
							temp_dist = dynamic_cast<nn::interfaces::SelectableInputInterface *>(inputs[inp]->From())->Accumulator_NaN_distance();
							if (temp_dist < shortest_dist) {
								shortest_dist = temp_dist;
								single_output_index = inp; 
							}
						}
					}
				}
			}
		}

		void NormalizeOwnLevel() {
			if constexpr (standalone) {
				throw std::exception("Logic error! NormalizeOwnLevel must not be used in standalone mode!");
			} else {
				if (std::isnan(single_output)) {
					void *largest_dist_conn = nullptr;
					float largest_dist = -1;
					float dist;
					for (const auto &inp : output_values) {
						dist = dynamic_cast<nn::interfaces::SelectableInputInterface *>(reinterpret_cast<nn::interfaces::CBI *>(inp.id)->From())->Accumulator_NaN_distance();
						if (dist > largest_dist) {
							largest_dist = dist;
							largest_dist_conn = inp.id;
						}
					}
					for (const auto &inp : output_values) {
						if (inp.id != largest_dist_conn) {
							dynamic_cast<nn::interfaces::SelectableInputInterface *>(reinterpret_cast<nn::interfaces::CBI *>(inp.id)->From())->Accumulator_make_NaN();
						}
					}
					single_output = reinterpret_cast<nn::interfaces::CBI *>(largest_dist_conn)->From()->OwnLevel();
				}
			}
		}

		void BatchAnalyzer_Init() {
			batch_analyzer_input_activate_min_distance.resize(inputs.size(), std::numeric_limits<float>::max());
		}

		void BatchAnalyzer_SetState(bool is_enabled) {
			batch_analyzer_disabled = !is_enabled;
		}

		// Usage Tip: 
		// It is assumed that you will view the values of the returned vector after each iteration (each input) and 
		// compare its minima with those stored on your side in order to save a reference to the data input at which the minimum fell.
		// If you see a value of 0, it means that the corresponding neuron has been activated at one of the inputs and 
		// additional correction of the relative value of this neuron is not required. Otherwise, you can "transfer" 
		// the most suitable data output to this neuron (the most suitable output is the one where the last decrease 
		// in the number occurred) using a special function.
		// Caution: the weights should not be adjusted during the batch, otherwise the statistics may not be representative!
		const std::vector<float> &BatchAnalyzer_GetInputsActivateMinDistance() const {
			return batch_analyzer_input_activate_min_distance;
		}

		unsigned BatchAnalyzer_GetCurrentInputIndex() const {
			return single_output_index;
		}

		void BatchAnalyzer_Reset() {
			std::fill(batch_analyzer_input_activate_min_distance.begin(), batch_analyzer_input_activate_min_distance.end(), std::numeric_limits<float>::max());
		}
		
		void BatchAnalyzer_Release() {
			std::vector<float>().swap(batch_analyzer_input_activate_min_distance);
		}

		void TransferCurrentInputToAnotherNeuron(unsigned current_neuron_index, unsigned new_neuron_index) {
			if (std::isnan(inputs[current_neuron_index]->From()->OwnLevel())) {
				throw std::exception("Logic error! TransferCurrentInputToAnotherNeuron source is NaN!");
			}
			if (!std::isnan(inputs[new_neuron_index]->From()->OwnLevel())) {
				throw std::exception("Logic error! TransferCurrentInputToAnotherNeuron destination is not NaN!");
			}
			dynamic_cast<nn::interfaces::SelectableInputInterface *>(inputs[current_neuron_index]->From())->Accumulator_make_NaN();
			dynamic_cast<nn::interfaces::SelectableInputInterface *>(inputs[new_neuron_index]->From())->Accumulator_make_unNaN();
		}

		float OwnLevel() override {
			return single_output;
		}

		bool IsCustomBackPropAvailable() override {
			const bool avail = !standalone;
			return avail;
		}

		const std::vector<Candidate> &RetriveCandidates() override {
			if constexpr (standalone) {
				throw std::exception("Logic error! RetriveCandidates must not be used in standalone mode!");
			} else {
				return output_values;
			}
		}

		void SelectBestCandidate(void *id, float error) override {
			if constexpr (standalone) {
				throw std::exception("Logic error! SelectBestCandidate must not be used in standalone mode!");
			} else {
				for (auto item : inputs) {
					if (item != id) {
						dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(false); // Set disposed to unused inputs
					} else {
						dynamic_cast<nn::interfaces::SelectableInputInterface *>(item->From())->SetBackPropEnabledState(true); // Set enabled to used inputs
					}
				}
				backprop_error_accumulator = error;
			}
		}

		void BackPropResetError() override {
			backprop_error_accumulator = 0;
		}

		void BackPropAccumulateError(float error) override {
			if constexpr (standalone) {
				backprop_error_accumulator += error;
			} else {
				throw std::exception("Logic error! BackPropAccumulateError must not be used in non-standalone mode!");
			}
		}

		float BackPropGetFinalError() override {
			return backprop_error_accumulator;
		}
	};
}
