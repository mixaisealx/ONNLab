#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicConvolutionI.h"
#include "BasicBackPropgI.h"
#include "CustomBackPropgI.h"
#include "BasicWghOptI.h"
#include "BasicLayerI.h"
#include "ErrorCalculatorI.h"

#include <vector>
#include <cmath>

namespace nn
{
	namespace errcalc
	{
		class ErrorCalcMSE : public interfaces::ErrorCalculatorI {
			std::vector<float> answers;
			float normalizer, *curr_ptr;
		public:
			ErrorCalcMSE(unsigned vector_size) {
				answers.reserve(vector_size);
				normalizer = 2.0f / vector_size;
				curr_ptr = nullptr;
			};
			void ProcessNeuronError(float neuron_out, float perfect_out) override {
				answers.push_back(neuron_out - perfect_out);
			}
			void DoCalc() override {
				curr_ptr = &answers[0];
			}
			float CalcLoss() override {
				float loss = 0;
				for (auto item : answers) {
					loss += item * item;
				}
				return loss / answers.size();
			}
			float GetNeuronPartialError() override {
				return *curr_ptr++ * normalizer;
			}
			void ResetState() override {
				answers.clear();
			}
		};

		class ErrorCalcSoftMAX : public interfaces::ErrorCalculatorI {
			std::vector<double> softmax;
			std::vector<float> perfect;
			double *softmax_ptr;
			float *perfect_ptr;
			float temperature_inv;
		public:
			const double MC_E = 2.7182818284590452353602874713527;

			ErrorCalcSoftMAX(unsigned vector_size, float temperature = 1.0f) {
				softmax.reserve(vector_size);
				perfect.reserve(vector_size);
				softmax_ptr = nullptr;
				perfect_ptr = nullptr;
				temperature_inv = 1.0f / temperature;
			};
			void ProcessNeuronError(float neuron_out, float perfect_out) override {
				perfect.push_back(perfect_out);
				softmax.push_back(std::pow(MC_E, neuron_out * temperature_inv));
			}
			void DoCalc() override {
				double summ = 0;
				for (auto val : softmax) summ += val;
				for (auto &val : softmax) val /= summ;

				softmax_ptr = &softmax[0];
				perfect_ptr = &perfect[0];
			}
			float CalcLoss() override {
				float loss = 0;
				for (size_t i = 0; i != softmax.size(); ++i) {
					loss += (perfect[i] * -static_cast<float>(std::log(softmax[i])));
				}
				return loss;
			}
			float GetNeuronPartialError() override {
				return static_cast<float>(*softmax_ptr++) - *perfect_ptr++;
			}
			void ResetState() override {
				softmax.clear();
				perfect.clear();
			}
			float Temperature() const {
				return 1.0f / temperature_inv;
			}
			void Temperature(float temperature) {
				temperature_inv = 1.0f / temperature;
			}
		};
	}
	

	class LearnGuiderFwBPg {
	public:
		LearnGuiderFwBPg(std::initializer_list<interfaces::BasicLayerInterface *> layers,
						  interfaces::ErrorCalculatorI *error_calculator, unsigned batch_size = 1U):
							layers(layers),
							ecalc(error_calculator),
							batch_size(batch_size), outs((*(layers.end() - 1))->Neurons()) {
			if (batch_size == 0) throw std::exception("batch_size cannot be zero!");
			weight_delta_bp.resize(batch_size);
			weight_delta_hdbp.resize(batch_size);
			weight_delta_optim.resize(batch_size);
		}

		LearnGuiderFwBPg(std::initializer_list<interfaces::BasicLayerInterface *> layers, unsigned batch_size = 1U):
			layers(layers),
			batch_size(batch_size), outs((*(layers.end() - 1))->Neurons()) {
			ecalc = nullptr;
			if (batch_size == 0) throw std::exception("batch_size cannot be zero!");
			weight_delta_bp.resize(batch_size);
			weight_delta_hdbp.resize(batch_size);
			weight_delta_optim.resize(batch_size);
		}

		void SetBatchSize(unsigned batch_size) {
			if (batch_size == 0) throw std::exception("batch_size cannot be zero!");
			weight_delta_bp.resize(batch_size);
			weight_delta_hdbp.resize(batch_size);
			weight_delta_optim.resize(batch_size);
			this->batch_size = batch_size;
		}

		unsigned GetBatchSize() const {
			return batch_size;
		}
		
		float FillupOutsError(const std::vector<float> &perfect_result, unsigned channel = 0, bool perform_loss_calculation = false) {
			if (!ecalc || perfect_result.size() != outs.size()) {
				throw std::exception("FillupOutsError is not ready!");
			}
			float loss = 0;
			ecalc->ResetState();

			const float *pptr = &perfect_result[0];
			for (auto nrn : outs) {
				if (!std::isnan(nrn->OwnLevel(channel))) {
					ecalc->ProcessNeuronError(nrn->OwnLevel(channel), *pptr++);
				} else { // NaN is not allowed if Custom backprop in standalone mode
					auto &candidates = dynamic_cast<nn::interfaces::CustomBackPropogableInterface *>(nrn)->RetriveCandidates();
					for (auto &cand : candidates) {
						ecalc->ProcessNeuronError(cand.value, *pptr);
					}
					++pptr;
				}
			}

			ecalc->DoCalc();
			if (perform_loss_calculation) {
				loss = ecalc->CalcLoss();
			}

			nn::interfaces::CustomBackPropogableInterface *special;
			for (auto nrn : outs) {
				special = dynamic_cast<nn::interfaces::CustomBackPropogableInterface *>(nrn);
				if (special == nullptr || !special->IsCustomBackPropAvailable()) {
					//dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn)->BackPropResetError(); //It is assumed that this function has already been called before.
					dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn)->BackPropAccumulateError(ecalc->GetNeuronPartialError(), channel);
				} else {
					unsigned idx_min = 0;
					float error_min = ecalc->GetNeuronPartialError();
					float current_error;
					for (unsigned idx = 1, end = special->RetriveCandidates().size(); idx < end; ++idx) {
						current_error = ecalc->GetNeuronPartialError();
						if (std::fabs(current_error) < std::fabs(error_min)) {
							error_min = current_error;
							idx_min = idx;
						}
					}
					special->SelectBestCandidate(special->RetriveCandidates()[idx_min].id, error_min);
				}
			}
			return loss;
		}

		const std::vector<interfaces::NeuronBasicInterface *> &GetOutputs() {
			return outs;
		}

		const std::vector<interfaces::BasicLayerInterface *> &GetLayers() {
			return layers;
		}

		void DoForward() {
			for (auto layer : layers) {
				if (!dynamic_cast<nn::interfaces::BackPropMetaLayerMark *>(layer)) { // Layer with neurons
					for (auto neuron : layer->Neurons()) {
						neuron->UpdateOwnLevel();
					}
				} else { // MetaLayer without standard neurons
					if (dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)) {
						dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)->PerformFullConvolution();
					}
				}
			}
		}

		void DoBackward() {
			float tval, tbp, toptim;

			nn::interfaces::NBI *nrn;
			// Advanced classes
			nn::interfaces::BasicBackPropogableInterface *basic_bp;
			nn::interfaces::MaccBackPropogableInterface *macc_bp;
			nn::interfaces::ZeroGradBackPropogableInterface *hidden_bp;

			for (auto citer = layers.rbegin(), eiter = layers.rend(); citer != eiter; ++citer) {
				if (!(*citer)->HasTrainable())
					break;

				if (!dynamic_cast<nn::interfaces::BackPropMetaLayerMark *>(*citer)) { // Layer with neurons
					for (auto neuron : (*citer)->Neurons()) {
						if (!dynamic_cast<nn::interfaces::InputNeuronI *>(neuron) && neuron->IsTrainable()) {
							basic_bp = dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(neuron);
							macc_bp = dynamic_cast<nn::interfaces::MaccBackPropogableInterface *>(neuron);
							hidden_bp = dynamic_cast<nn::interfaces::ZeroGradBackPropogableInterface *>(neuron);

							for (unsigned channel = 0; channel != batch_size; ++channel) {
								if (macc_bp) {
									tval = macc_bp->SurrogateAccumulatorValue(channel); // Seems like function isn't monotonic, so derivative must be computed from source value
								} else {
									tval = neuron->OwnLevel(channel);
								}

								if (hidden_bp) {
									// Neuron has a hidden backprop coefficient
									tbp = toptim = hidden_bp->BackPropGetFinalError(channel);
									tbp *= neuron->ActivationFunctionDerivative(tval);
									toptim *= hidden_bp->HiddenActivationFunctionDerivative(tval, toptim); // weight_delta_optim = hidden_bp->BackPropGetFinalError()
									weight_delta_hdbp[channel] = toptim * hidden_bp->BackPropErrorFactor(tval);
									weight_delta_optim[channel] = toptim;
									weight_delta_bp[channel] = tbp;
								} else {
									weight_delta_hdbp[channel] = weight_delta_optim[channel] = weight_delta_bp[channel] = basic_bp->BackPropGetFinalError(channel) * neuron->ActivationFunctionDerivative(tval);
								}
							}
							basic_bp->BackPropResetError();

							for (auto inpconn : neuron->InputConnections()) {
								nrn = inpconn->From();
								if (nrn->IsTrainable()) {
									hidden_bp = dynamic_cast<nn::interfaces::ZeroGradBackPropogableInterface *>(nrn);
									tval = inpconn->Weight();
									if (hidden_bp) {
										for (unsigned channel = 0; channel != batch_size; ++channel) {
											hidden_bp->BackPropAccumulateError(weight_delta_hdbp[channel] * tval, channel);
										}
									} else {
										basic_bp = dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn);
										for (unsigned channel = 0; channel != batch_size; ++channel) {
											basic_bp->BackPropAccumulateError(weight_delta_bp[channel] * tval, channel);
										}
									}
								}
								// Gradient averaging
								tval = 0.0f;
								for (unsigned channel = 0; channel != batch_size; ++channel) {
									tval += weight_delta_optim[channel] * nrn->OwnLevel(channel);
								}
								tval /= batch_size;
								// Weight updating
								dynamic_cast<nn::interfaces::BasicWeightOptimizableInterface *>(inpconn)->WeightOptimDoUpdate(tval);
							}
						}
					}
				} else { // MetaLayer without standard neurons
					if (dynamic_cast<nn::interfaces::BasicConvolutionI *>(*citer)) {
						dynamic_cast<nn::interfaces::BasicConvolutionI *>(*citer)->BackPropagateFullConvolution();
					}
				}
			}
		}

		struct BackwardSpecialExtra {
			struct LayerOverride {
				unsigned short layer_id;
				bool update_weights, cleanup_grads;
			};

			void *convolution_extra = nullptr;
			std::vector<LayerOverride> layers_overrides;
		};
		void DoBackwardSpecial(bool update_weights, bool cleanup_grads, BackwardSpecialExtra *extra) {
			float tval, tbp, toptim;

			std::vector<BackwardSpecialExtra::LayerOverride> overrides;

			if (extra) {
				overrides = extra->layers_overrides;
				// Sort desc
				std::sort(overrides.begin(), overrides.end(), [](const BackwardSpecialExtra::LayerOverride &a, const BackwardSpecialExtra::LayerOverride &b) {
					return a.layer_id > b.layer_id;
				});
			}
			auto overrides_current = overrides.begin();
			auto overrides_end = overrides.end();

			nn::interfaces::NBI *nrn;
			// Advanced classes
			nn::interfaces::BasicBackPropogableInterface *basic_bp;
			nn::interfaces::MaccBackPropogableInterface *macc_bp;
			nn::interfaces::ZeroGradBackPropogableInterface *hidden_bp;

			bool update_weights_bak = update_weights, cleanup_grads_bak = cleanup_grads;

			unsigned short layer_id = static_cast<unsigned short>(layers.size() - 1);
			for (auto citer = layers.rbegin(), eiter = layers.rend(); citer != eiter; ++citer, --layer_id) {
				if (overrides_current != overrides_end && overrides_current->layer_id == layer_id) {
					update_weights = overrides_current->update_weights;
					cleanup_grads = overrides_current->cleanup_grads;
					++overrides_current;
					while (overrides_current != overrides_end && overrides_current->layer_id == layer_id) {
						++overrides_current;
					}
				} else {
					update_weights = update_weights_bak;
					cleanup_grads = cleanup_grads_bak;
				}

				if (!(*citer)->HasTrainable())
					break;

				if (!dynamic_cast<nn::interfaces::BackPropMetaLayerMark *>(*citer)) { // Layer with neurons
					for (auto neuron : (*citer)->Neurons()) {
						if (!dynamic_cast<nn::interfaces::InputNeuronI *>(neuron) && neuron->IsTrainable()) {
							basic_bp = dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(neuron);
							macc_bp = dynamic_cast<nn::interfaces::MaccBackPropogableInterface *>(neuron);
							hidden_bp = dynamic_cast<nn::interfaces::ZeroGradBackPropogableInterface *>(neuron);

							for (unsigned channel = 0; channel != batch_size; ++channel) {
								if (macc_bp) {
									tval = macc_bp->SurrogateAccumulatorValue(channel); // Seems like function isn't monotonic, so derivative must be computed from source value
								} else {
									tval = neuron->OwnLevel(channel);
								}

								if (hidden_bp) {
									// Neuron has a hidden backprop coefficient
									tbp = toptim = hidden_bp->BackPropGetFinalError(channel);
									tbp *= neuron->ActivationFunctionDerivative(tval);
									toptim *= hidden_bp->HiddenActivationFunctionDerivative(tval, toptim); // weight_delta_optim = hidden_bp->BackPropGetFinalError()
									weight_delta_hdbp[channel] = toptim * hidden_bp->BackPropErrorFactor(tval);
									weight_delta_optim[channel] = toptim;
									weight_delta_bp[channel] = tbp;
								} else {
									weight_delta_hdbp[channel] = weight_delta_optim[channel] = weight_delta_bp[channel] = basic_bp->BackPropGetFinalError(channel) * neuron->ActivationFunctionDerivative(tval);
								}
							}

							if (cleanup_grads) {
								basic_bp->BackPropResetError();
							}

							for (auto inpconn : neuron->InputConnections()) {
								nrn = inpconn->From();
								if (nrn->IsTrainable()) {
									hidden_bp = dynamic_cast<nn::interfaces::ZeroGradBackPropogableInterface *>(nrn);
									tval = inpconn->Weight();
									if (hidden_bp) {
										for (unsigned channel = 0; channel != batch_size; ++channel) {
											hidden_bp->BackPropAccumulateError(weight_delta_hdbp[channel] * tval, channel);
										}
									} else {
										basic_bp = dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn);
										for (unsigned channel = 0; channel != batch_size; ++channel) {
											basic_bp->BackPropAccumulateError(weight_delta_bp[channel] * tval, channel);
										}
									}
								}

								if (update_weights) {
									// Gradient averaging
									tval = 0.0f;
									for (unsigned channel = 0; channel != batch_size; ++channel) {
										tval += weight_delta_optim[channel] * nrn->OwnLevel(channel);
									}
									tval /= batch_size;
									// Weight updating
									dynamic_cast<nn::interfaces::BasicWeightOptimizableInterface *>(inpconn)->WeightOptimDoUpdate(tval);
								}
							}
						}
					}
				} else { // MetaLayer without standard neurons
					if (dynamic_cast<nn::interfaces::BasicConvolutionI *>(*citer)) {
						dynamic_cast<nn::interfaces::BasicConvolutionI *>(*citer)->BackPropagateFullConvolutionSpecial(update_weights, cleanup_grads, (extra ? extra->convolution_extra : nullptr));
					}
				}
			}
		}

	private:
		// Operational temp vars
		std::vector<float> weight_delta_bp; // The back propagation step with visible error
		std::vector<float> weight_delta_hdbp; // The back propagation step with hidden error
		std::vector<float> weight_delta_optim; // The weight optimization step

		// Inner variables
		std::vector<interfaces::BasicLayerInterface *> layers;
		const std::vector<interfaces::NeuronBasicInterface *> &outs;
		interfaces::ErrorCalculatorI *ecalc;
		unsigned batch_size;
	};
}
