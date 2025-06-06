#pragma once
#include "LearnGuiderFwBPg.h"
#include "AtomicSpinlock.h"

#include <barrier>

namespace nn
{
	class LearnGuiderFwBPgThreadAble {
	public:
		LearnGuiderFwBPgThreadAble(std::initializer_list<interfaces::BasicLayerInterface *> layers, unsigned batch_size, unsigned threads_count): threads_count(threads_count),
							layers(layers),
							batch_size(batch_size), outs((*(layers.end() - 1))->Neurons()) {
			if (batch_size == 0) throw std::exception("batch_size cannot be zero!");

			threads_barrier = new std::barrier<>(threads_count);

			cache_cnt_forward = 0;
			cache_cnt_all = 3;
			for (auto layer : layers) {
				if (dynamic_cast<nn::interfaces::BackPropMetaLayerMark *>(layer)) {
					if (dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)) {
						auto lyr = dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer);
						lyr->SetThreadsCount(threads_count);
						cache_cnt_forward = std::max(cache_cnt_forward, lyr->GetRequiredCachesCount(false));
						cache_cnt_all = std::max(cache_cnt_all, lyr->GetRequiredCachesCount(true));
					}
				}
			}
			cache_cnt_all = std::max(cache_cnt_all, cache_cnt_forward);
		}

		~LearnGuiderFwBPgThreadAble() {
			delete threads_barrier;
		}

		unsigned GetRequiredCachesCount(bool for_all = true) const {
			return (for_all ? cache_cnt_all : cache_cnt_forward);
		}

		unsigned GetThreadsCount() const {
			return threads_count;
		}

		void SetThreadsCount(unsigned threads_count) {
			if (!threads_count)
				throw std::exception("Zero threads count!");

			if (this->threads_count != threads_count) {
				this->threads_count = threads_count;

				delete threads_barrier;
				threads_barrier = new std::barrier<>(threads_count);

				for (auto layer : layers) {
					if (dynamic_cast<nn::interfaces::BackPropMetaLayerMark *>(layer)) {
						if (dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)) {
							dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)->SetThreadsCount(threads_count);
						}
					}
				}
			}
		}

		void SetBatchSize(unsigned batch_size) {
			if (!batch_size) throw std::exception("batch_size cannot be zero!");
			this->batch_size = batch_size;
		}

		unsigned GetBatchSize() const {
			return batch_size;
		}
		
		inline float FillupOutsError(unsigned worker_id, interfaces::ErrorCalculatorI *ecalc, const std::vector<std::vector<float>> &perfect_result, bool perform_loss_calculation = false) {
			float loss = 0.0f;

			for (unsigned channel = worker_id; channel < batch_size; channel += threads_count) {
				ecalc->ResetState();

				const float *pptr = &perfect_result[channel][0];
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
					loss += ecalc->CalcLoss();
				}

				nn::interfaces::CustomBackPropogableInterface *special;
				for (auto nrn : outs) {
					special = dynamic_cast<nn::interfaces::CustomBackPropogableInterface *>(nrn);
					if (special == nullptr || !special->IsCustomBackPropAvailable()) {
						InterlockedBackPropAccumulateError(dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn), ecalc->GetNeuronPartialError(), channel);
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
						locker.lock();
						special->SelectBestCandidate(special->RetriveCandidates()[idx_min].id, error_min);
						locker.unlock();
					}
				}
			}

			threads_barrier->arrive_and_wait();

			return loss;
		}

		const std::vector<interfaces::NeuronBasicInterface *> &GetOutputs() {
			return outs;
		}

		const std::vector<interfaces::BasicLayerInterface *> &GetLayers() {
			return layers;
		}

		inline void WorkerDoForward(unsigned worker_id, std::vector<float> caches[]) {
			for (auto layer : layers) {
				if (!dynamic_cast<nn::interfaces::BackPropMetaLayerMark *>(layer)) { // Layer with neurons
					auto &neurolist = layer->Neurons();
					for (unsigned mi = worker_id; mi < neurolist.size(); mi += threads_count) {
						neurolist[mi]->UpdateOwnLevel();
					}
				} else { // MetaLayer without standard neurons
					if (dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)) {
						dynamic_cast<nn::interfaces::BasicConvolutionI *>(layer)->PerformPartialConvolution(worker_id, caches);
					}
				}
				threads_barrier->arrive_and_wait();
			}
		}

		inline void WorkerDoBackward(unsigned worker_id, std::vector<float> caches[]) {
			std::vector<float> &weight_delta_bp = caches[0];
			std::vector<float> &weight_delta_optim = caches[1];
			std::vector<float> &weight_delta_hdbp = caches[2];

			if (weight_delta_bp.size() < batch_size)
				weight_delta_bp.resize(batch_size);
			if (weight_delta_optim.size() < batch_size)
				weight_delta_optim.resize(batch_size);
			if (weight_delta_hdbp.size() < batch_size)
				weight_delta_hdbp.resize(batch_size);
			
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
					auto &neurolist = (*citer)->Neurons();
					for (unsigned mi = worker_id; mi < neurolist.size(); mi += threads_count) {
						auto &neuron = neurolist[mi];

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
											InterlockedBackPropAccumulateErrorZG(hidden_bp, weight_delta_hdbp[channel] * tval, channel);
										}
									} else {
										basic_bp = dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(nrn);
										for (unsigned channel = 0; channel != batch_size; ++channel) {
											InterlockedBackPropAccumulateError(basic_bp, weight_delta_bp[channel] * tval, channel);
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
						dynamic_cast<nn::interfaces::BasicConvolutionI *>(*citer)->BackPropagatePartialConvolution(worker_id, caches);
					}
				}

				threads_barrier->arrive_and_wait();
			}
		}

	private:
		inline void InterlockedBackPropAccumulateError(interfaces::BasicBackPropogableInterface *nrn, float error, unsigned channel) {
			locker.lock();
			nrn->BackPropAccumulateError(error, channel);
			locker.unlock();
		}
		inline void InterlockedBackPropAccumulateErrorZG(interfaces::ZeroGradBackPropogableInterface *nrn, float error, unsigned channel) {
			locker.lock();
			nrn->BackPropAccumulateError(error, channel);
			locker.unlock();
		}

		// Inner variables
		std::vector<interfaces::BasicLayerInterface *> layers;
		const std::vector<interfaces::NeuronBasicInterface *> &outs;
		unsigned batch_size, threads_count;
		unsigned cache_cnt_forward, cache_cnt_all;
		
		std::barrier<> *threads_barrier;
		AtomicSpinlock locker;
	};
}
