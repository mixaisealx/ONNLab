#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicBackPropgI.h"
#include "CustomBackPropgI.h"
#include "BasicWghOptI.h"
#include "BasicLayerI.h"
#include "OutsErrorSetterI.h"

#include <vector>
#include <math.h>

namespace nn
{
	class BasicOutsErrorSetter : public interfaces::OutsErrorSetterI {
	public:
		class ErrorCalcMSE : public ErrorCalculatorI {
			std::vector<float> answers;
			float normalizer, *curr_ptr;
		public:
			ErrorCalcMSE() {
				normalizer = 0;
				curr_ptr = nullptr;
			};
			~ErrorCalcMSE() = default;
			void InitState(unsigned vector_size) override {
				answers.reserve(vector_size);
				normalizer = 2.0f / vector_size;
			}
			void ProcessNeuronError(float neuron_out, float perfect_out) override {
				//answers.push_back((neuron_out - perfect_out) * normalizer);
				answers.push_back(neuron_out - perfect_out);
			}
			void DoCalc() override {
				curr_ptr = &answers[0];
			}
			float GetNeuronPartialError() override {
				return *curr_ptr++ * normalizer;
			}
			void ResetState() override {
				answers.clear();
			}
		};

		class ErrorCalcSoftMAX : public ErrorCalculatorI {
			std::vector<float> softmax;
			std::vector<float> perfect;
			float *softmax_ptr, *perfect_ptr;
		public:
			const float MC_E = 2.71828182f;

			ErrorCalcSoftMAX() {
				softmax_ptr = perfect_ptr = nullptr;
			};
			~ErrorCalcSoftMAX() = default;
			void InitState(unsigned vector_size) override {
				softmax.reserve(vector_size);
				perfect.reserve(vector_size);
			}
			void ProcessNeuronError(float neuron_out, float perfect_out) override {
				perfect.push_back(perfect_out);
				softmax.push_back(powf(MC_E, neuron_out));
			}
			void DoCalc() override {
				float summ = 0;
				for (auto val : softmax) summ += val;
				for (auto &val : softmax) val /= summ;

				softmax_ptr = &softmax[0];
				perfect_ptr = &perfect[0];
			}
			float GetNeuronPartialError() override {
				return *softmax_ptr++ - *perfect_ptr++;
			}
			void ResetState() override {
				softmax.clear();
				perfect.clear();
			}
		};

		BasicOutsErrorSetter(ErrorCalculatorI *ecalc, unsigned int vector_size):ecalc(ecalc) {
			ecalc->InitState(vector_size);
		}

		void FillupError(std::vector<interfaces::NeuronBasicInterface *> &outputs, const std::vector<float> &perfect_result) override {
			ecalc->ResetState();

			const float *pptr = &perfect_result[0];
			for (auto nrn : outputs) {
				if (!std::isnan(nrn->OwnLevel())) {
					ecalc->ProcessNeuronError(nrn->OwnLevel(), *pptr++);
				} else { // NaN is not allowed if Custom backprop in standalone mode
					auto &candidates = dynamic_cast<nn::interfaces::CustomBackPropogableInterface *>(nrn)->RetriveCandidates();
					for (auto &cand : candidates) {
						ecalc->ProcessNeuronError(cand.value, *pptr);
					}
					++pptr;
				}
			}
			
			ecalc->DoCalc();

			nn::interfaces::CustomBackPropogableInterface *special;
			for (auto nrn : outputs) {
				special = dynamic_cast<nn::interfaces::CustomBackPropogableInterface *>(nrn);
				if (special == nullptr || !special->IsCustomBackPropAvailable()) {
					//dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn)->BackPropResetError(); //It is assumed that this function has already been called before.
					dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn)->BackPropAccumulateError(ecalc->GetNeuronPartialError());
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
		}

	private:
		ErrorCalculatorI *ecalc;
	};

	class FwdBackPropGuider {
	public:
		FwdBackPropGuider(std::initializer_list<interfaces::BasicLayerInterface *> layers,
						  interfaces::OutsErrorSetterI *esetter):layers(layers), errset(esetter) {
			for (auto neuron : this->layers.back()->Neurons()) {
				outs.push_back(neuron);
			}
		}

		void DoForward() {
			for (auto layer : layers) {
				for (auto neuron : layer->Neurons()) {
					neuron->UpdateOwnLevel();
				}
			}
		}

		void FillupOutsError(const std::vector<float> &perfect_result) {
			if (perfect_result.size() != outs.size()) {
				throw std::exception();
			}
			errset->FillupError(outs, perfect_result);
		}

		void DoBackward() {
			float weight_delta; // The back propagation step

			// Advanced classes
			nn::interfaces::MaccBackPropogableInterface *macc_bp;

			for (auto citer = layers.rbegin(), eiter = layers.rend(); citer != eiter; ++citer) {
				if (!(*citer)->HasTrainable()) 
					break;

				for (auto neuron : (*citer)->Neurons()) {
					if (neuron->IsTrainable()) {
						macc_bp = dynamic_cast<nn::interfaces::MaccBackPropogableInterface *>(neuron);
						// Checking for advanced classes
						if (macc_bp) { 
							// Seems like function isn't linear, so derivative must be computed from source value
							weight_delta = macc_bp->BackPropGetFinalError() * neuron->ActivationFunctionDerivative(macc_bp->OwnAccumulatorValue());
						} else {
							weight_delta = dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(neuron)->BackPropGetFinalError() * neuron->ActivationFunctionDerivative(neuron->OwnLevel());
						}
						dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(neuron)->BackPropResetError();

						for (auto inpconn : neuron->InputConnections()) {
							if (inpconn->From()->IsTrainable()) {
								dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(inpconn->From())->BackPropAccumulateError(weight_delta * inpconn->Weight());
							}
							dynamic_cast<nn::interfaces::BasicWeightOptimizableInterface *>(inpconn)->WeightOptimDoUpdate(weight_delta);
						}
					}
				}
			}
		}

	private:
		std::vector<interfaces::BasicLayerInterface *> layers;
		std::vector<interfaces::NeuronBasicInterface *> outs;
		interfaces::OutsErrorSetterI *errset;
	};
}
