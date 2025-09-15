#pragma once

#include "OptimizerAdam.h"
#include "LearnGuiderFwBPgThreadAble.h"
#include "NetQualityCalcUtils.h"

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>
#include <barrier>
#include <atomic>
#include <stdexcept>


namespace nn::reverse
{
	// Settings for Carlini-Wagner L2 Attack
	struct CarliniWagnerL2Params {
		CarliniWagnerL2Params(
			float box_min = 0.0f, // minimum pixel value
			float box_max = 1.0f, // maximum pixel value
			float confidence = 0.0f, // minimum required difference between the classes as a result
			bool allow_early_stop = true, // whether to stop early if no progress
			unsigned short binary_search_steps = 9, // binary search iterations for trade-off constant
			unsigned max_iterations = 10000, // max optimization iterations
			float l2_grad_influence = 1.0f, // learning-rate-like; if smaller, the less influence the L2 norm of the difference between the original and the adversal inputs has
			float early_stop_severity = 0.9999f, // severity factor of loss changes for continuation, midpoint = 1.0, lower - more severe, higher - less severe
			unsigned early_stop_chances_count = 5, // the number of "second" chances so loss can decrease sufficiently and not perform a early stop
			float learning_rate = 1e-2f, // Adam learning rate
			float loss1_scale_init = 1e-3f // initial trade-off constant for loss1
		): binary_search_steps(binary_search_steps),
			max_iterations(max_iterations),
			confidence(confidence),
			early_stop_chances_count(early_stop_chances_count),
			state_printing_period(100),
			learning_rate(learning_rate),
			loss1_scale_init(loss1_scale_init),
			early_stop_severity(early_stop_severity),
			l2_grad_influence(l2_grad_influence),
			box_min(box_min),
			box_max(box_max),
			allow_early_stop(allow_early_stop),
			StatePrinter(nullptr) {}
		unsigned short binary_search_steps; // binary search iterations for trade-off constant
		unsigned max_iterations; // max optimization iterations
		unsigned early_stop_chances_count; // the number of "second" chances so loss can decrease sufficiently and not perform a early stop
		unsigned state_printing_period; // the iteration period of state output
		float confidence; // minimum required difference between the classes as a result
		float learning_rate; // Adam learning rate
		float loss1_scale_init; // initial trade-off constant for loss1
		float early_stop_severity; // severity factor of loss changes for continuation, midpoint = 1.0, lower - more severe, higher - less severe
		float l2_grad_influence; // learning-rate-like; if smaller, the less influence the L2 norm of the difference between the original and the adversal inputs has
		float box_min; // minimum pixel value
		float box_max; // maximum pixel value
		bool allow_early_stop; // whether to stop early if no progress
		using StatePrinterT = void(*)(unsigned short binstep, unsigned iteration, float loss);
		StatePrinterT StatePrinter;
	};

	// Carlini-Wagner L2 Attack
	class CarliniWagnerL2 {
		LearnGuiderFwBPg &model;
		unsigned batch_size, max_iterations, early_stop_hits_max, state_printing_period;
		float confidence, learning_rate, loss1_scale_init, early_stop_severity, l2_grad_influence;
		float box_min, box_max, box_pxl_plus, box_pxl_mul, box_pxl_div;
		unsigned short binary_search_steps;
		bool is_targeted;
		CarliniWagnerL2Params::StatePrinterT StatePrinter;

		const std::vector<interfaces::NBI *> &nninputs, &nnoutputs;
		std::vector<interfaces::InputNeuronI *> nninputs_ip;
		std::vector<interfaces::BasicBackPropogableInterface *> nninputs_bp, nnoutputs_bp;

		// Compute the squared L2 (Euclidean) distance
		static inline float l2_norm2(const std::vector<float> &a, const std::vector<float> &b) {
			float sum = 0.0f;
			float diff;
			for (unsigned i = 0; i != a.size(); ++i) {
				diff = a[i] - b[i];
				sum += diff * diff;
			}
			return sum;
		}

		// Project to tanh space
		inline float space_proj_to_tanh(float x) const {
			return std::atanh(((x * 0.9999999f) - box_pxl_plus) * box_pxl_div);
		}

		// Project to atanh space (pixel space)
		inline float space_proj_to_pixel(float x) const {
			return std::tanh(x) * box_pxl_mul + box_pxl_plus;
		}

		inline void LoadToNN(std::vector<std::vector<float>> &modifier, std::vector<std::vector<float>> &current_attack, const std::vector<std::vector<float>> &tanh_inputs) {
			for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
				auto &modrow = modifier[batch_idx];
				auto &tanrow = tanh_inputs[batch_idx];
				auto &catrow = current_attack[batch_idx];
				float tmp;
				for (unsigned inp_idx = 0; inp_idx != nninputs_ip.size(); ++inp_idx) {
					catrow[inp_idx] = tmp = space_proj_to_pixel(modrow[inp_idx] + tanrow[inp_idx]);
					nninputs_ip[inp_idx]->SetOwnLevel(tmp, batch_idx);
				}
			}
		}

		CarliniWagnerL2(const CarliniWagnerL2 &) = delete;
		CarliniWagnerL2 &operator=(const CarliniWagnerL2 &) = delete;
	public:

		CarliniWagnerL2(LearnGuiderFwBPg &model,
						unsigned batch_size,
						bool is_targeted = true):CarliniWagnerL2(model, CarliniWagnerL2Params(), batch_size, is_targeted) {}


		CarliniWagnerL2(LearnGuiderFwBPg &model,
						CarliniWagnerL2Params params,
						unsigned batch_size,
						bool is_targeted = true)
			: model(model), nninputs(model.GetLayers()[0]->Neurons()), nnoutputs(model.GetOutputs()),
			batch_size(batch_size),
			max_iterations(params.max_iterations),
			early_stop_hits_max(params.early_stop_chances_count),
			state_printing_period(params.state_printing_period),
			confidence(params.confidence),
			learning_rate(params.learning_rate),
			loss1_scale_init(params.loss1_scale_init),
			early_stop_severity(params.allow_early_stop ? params.early_stop_severity : std::numeric_limits<float>::infinity()),
			l2_grad_influence(params.l2_grad_influence * 2.0f),
			box_min(params.box_min),
			box_max(params.box_max),
			binary_search_steps(params.binary_search_steps),
			is_targeted(is_targeted), 
			StatePrinter(params.StatePrinter) {
			nninputs_bp.reserve(nninputs.size());
			nninputs_ip.reserve(nninputs.size());
			for (auto elem : nninputs) {
				if (!dynamic_cast<nn::interfaces::InputNeuronI *>(elem) || !dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The input layer must consist of neurons based on interfaces::BasicBackPropogableInterface!");
				}
				nninputs_ip.push_back(dynamic_cast<interfaces::InputNeuronI *>(elem));
				nninputs_bp.push_back(dynamic_cast<interfaces::BasicBackPropogableInterface *>(elem));
			}

			nnoutputs_bp.reserve(nnoutputs.size());
			for (auto elem : nnoutputs) {
				if (!dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The output layer must consist of neurons based on interfaces::BasicBackPropogableInterface!");
				}
				nnoutputs_bp.push_back(dynamic_cast<interfaces::BasicBackPropogableInterface *>(elem));
			}

			// factors to map between tanh-space and original pixel bounds
			box_pxl_plus = (box_max + box_min) / 2.0f;
			box_pxl_mul = (box_max - box_min) / 2.0f;
			box_pxl_div = 1.0f / box_pxl_mul;
		}

		// Attack a single batch of inputs
		std::vector<std::vector<float>> RunAttack(const std::vector<std::vector<float>> &input_batch, const std::vector<unsigned short> &target_classes, unsigned *overall_iterations_count_return = nullptr) {
			std::vector<std::vector<float>> best_attack(batch_size); // For RVO

			if (input_batch.size() != batch_size || target_classes.size() != batch_size) {
				throw std::runtime_error("input_batch size != batch_size or target_classes size != batch_size");
			}

			std::vector<std::vector<float>> current_attack(batch_size, std::vector<float>(nninputs.size()));

			std::vector<std::vector<float>> tanh_inputs(batch_size, std::vector<float>(nninputs.size()));
			// Convert inputs into tanh-space so that any modifier remains in valid pixel range
			for (int i = 0; i != batch_size; ++i) {
				for (int j = 0; j != nninputs.size(); ++j) {
					tanh_inputs[i][j] = space_proj_to_tanh(input_batch[i][j]);
				}
			}

			nn::optimizers::Adam optimizer(learning_rate);
			std::vector<std::vector<nn::optimizers::Adam::State>> optimizer_states(batch_size, std::vector<nn::optimizers::Adam::State>(nninputs.size()));


			// Setup binary search bounds for the trade-off constant c
			std::vector<float> loss_scale_lower(batch_size, 0.0f),
				loss_scale_upper(batch_size, 1e10f),
				loss_scale(batch_size, loss1_scale_init);

			// To store best adversarial example per image
			std::vector<float> best_l2(batch_size, std::numeric_limits<float>::max());

			std::vector<std::vector<float>> modifier(batch_size, std::vector<float>(nninputs.size()));

			unsigned overall_iters = 0;

			// Outer loop: binary search over loss1_scale
			for (unsigned short bins = 0; bins != binary_search_steps; ++bins) {
				// Initialize modifier (the variable we optimize) to zero
				for (auto &mod : modifier) {
					std::fill(mod.begin(), mod.end(), 0.0f);
				}

				// Reset Adam
				for (auto &bstt : optimizer_states) {
					for (auto &stt : bstt) {
						optimizer.Reset(&stt);
					}
				}

				// Track previous loss for early abort
				float loss_prev = std::numeric_limits<float>::max();
				float loss = 0.0f;
				unsigned loss_hits = 0;


				// Inner loop: minimization of loss1*c + loss2 (L2)
				for (unsigned iteration = 0;; ++iteration) {
					// Building & loading adversarial example in pixel-space
					LoadToNN(modifier, current_attack, tanh_inputs);

					// Forward propagation for batch
					model.DoForward();


					if (iteration == max_iterations) {
						overall_iters += iteration;
						break;
					}

					loss = 0.0f;
					// Compute error & classification loss
					for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
						unsigned short target_class = target_classes[batch_idx];
						//    real = logit of target class
						float real = nnoutputs[target_class]->OwnLevel(batch_idx);
						//    other = max logit of non-targets
						float other = -std::numeric_limits<float>::infinity();
						unsigned short other_class = std::numeric_limits<unsigned short>::max();
						{
							float tval;
							for (unsigned short k = 0; k != nnoutputs.size(); ++k) {
								if (k != target_class) {
									tval = nnoutputs[k]->OwnLevel(batch_idx);
									if (tval > other) {
										other = tval;
										other_class = k;
									}
								}
							}
						}

						// hinge loss
						float loss1;
						if (is_targeted) {
							loss1 = std::max(0.0f, other - real + confidence);
							if (loss1 > 0.0f) {
								nnoutputs_bp[target_class]->BackPropAccumulateError(loss_scale[batch_idx], batch_idx); // +1.0f * loss_scale[batch_idx]
								nnoutputs_bp[other_class]->BackPropAccumulateError(-loss_scale[batch_idx], batch_idx); // -1.0f * loss_scale[batch_idx]
							}
						} else {
							loss1 = std::max(0.0f, real - other + confidence);
							if (loss1 > 0.0f) {
								nnoutputs_bp[other_class]->BackPropAccumulateError(loss_scale[batch_idx], batch_idx); // +1.0f * loss_scale[batch_idx]
								nnoutputs_bp[target_class]->BackPropAccumulateError(-loss_scale[batch_idx], batch_idx); // -1.0f * loss_scale[batch_idx]
							}
						}

						// total loss
						loss += loss1 * loss_scale[batch_idx] + l2_norm2(current_attack[batch_idx], input_batch[batch_idx]);
					}

					// Optional early abort if loss stops improving
					if (loss < loss_prev * early_stop_severity) {
						loss_prev = loss; // update baseline
						loss_hits = 0;
					} else {
						// no significant improvement
						if (++loss_hits > early_stop_hits_max) {
							for (auto &nrn : nnoutputs_bp) {
								nrn->BackPropResetError();
							}
							overall_iters += iteration;
							break;
						}
					}

					// Backward propagation for batch
					model.DoBackward();

					// Modifier optimization
					for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
						auto &inprow = input_batch[batch_idx];
						auto &catrow = current_attack[batch_idx];
						auto &modrow = modifier[batch_idx];
						auto &optrow = optimizer_states[batch_idx];
						float grad;
						for (unsigned inp_idx = 0; inp_idx != nninputs_bp.size(); ++inp_idx) {
							grad = nninputs_bp[inp_idx]->BackPropGetFinalError(batch_idx);
							grad += l2_grad_influence * (inprow[inp_idx] - catrow[inp_idx]); // Applying L2 norm grad: 2.0 * (inprow[inp_idx] - catrow[inp_idx])
							grad *= box_pxl_div; // Applying scale to convert form pixel to tanh space
							modrow[inp_idx] += optimizer.CalcDelta(grad, &optrow[inp_idx]);
						}
					}

					if (StatePrinter && !(iteration % state_printing_period)) {
						StatePrinter(bins, iteration, loss);
					}

					// Reset inputs grads
					for (auto &nrn : nninputs_bp) {
						nrn->BackPropResetError();
					}
				}

				// Build final adversarial examples and update bests, do binary search step
				for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
					unsigned short tgt = target_classes[batch_idx];
					unsigned short pred = nn::netquality::NeuroArgmax(nnoutputs, batch_idx);
					bool success = is_targeted ? (pred == tgt) : (pred != tgt);

					// If successful
					if (success) {
						if (best_attack[batch_idx].empty()) {
							best_attack[batch_idx].resize(nninputs.size());
						}

						float l2 = l2_norm2(current_attack[batch_idx], input_batch[batch_idx]);
						if (l2 < best_l2[batch_idx]) { // If smaller L2 than previous, record as best
							best_l2[batch_idx] = l2;
							best_attack[batch_idx] = current_attack[batch_idx];
						}

						// if attack succeeded, try smaller loss_scale
						loss_scale_upper[batch_idx] = std::min(loss_scale_upper[batch_idx], loss_scale[batch_idx]);
						loss_scale[batch_idx] = (loss_scale_lower[batch_idx] + loss_scale_upper[batch_idx]) / 2.0f;
					} else {
						// if failed, increase loss_scale (or binary search if upper bound known)
						loss_scale_lower[batch_idx] = std::max(loss_scale_lower[batch_idx], loss_scale[batch_idx]);
						if (loss_scale_upper[batch_idx] < 1e9f)
							loss_scale[batch_idx] = (loss_scale_lower[batch_idx] + loss_scale_upper[batch_idx]) / 2.0f;
						else
							loss_scale[batch_idx] *= 10.0f;
					}
				}

				if (StatePrinter) {
					StatePrinter(bins, std::numeric_limits<unsigned>::max(), loss);
				}
			}

			if (overall_iterations_count_return) {
				*overall_iterations_count_return = overall_iters;
			}

			// Return the best adversarial examples found
			return best_attack;
		}

	};

	// Carlini-Wagner L2 Attack supporting multi-thread processing
	class CarliniWagnerL2ThreadAble {
		LearnGuiderFwBPgThreadAble &model;
		unsigned threads_count, batch_size, max_iterations, early_stop_hits_max, state_printing_period;
		float confidence, learning_rate, loss1_scale_init, early_stop_severity, l2_grad_influence;
		float box_min, box_max, box_pxl_plus, box_pxl_mul, box_pxl_div;
		unsigned short binary_search_steps;
		bool is_targeted, allow_early_stop;
		CarliniWagnerL2Params::StatePrinterT StatePrinter;

		const std::vector<interfaces::NBI *> &nninputs, &nnoutputs;
		std::vector<interfaces::InputNeuronI *> nninputs_ip;
		std::vector<interfaces::BasicBackPropogableInterface *> nninputs_bp, nnoutputs_bp;

		// Compute the squared L2 (Euclidean) distance
		static inline float l2_norm2(const std::vector<float> &a, const std::vector<float> &b) {
			float sum = 0.0f;
			float diff;
			for (unsigned i = 0; i != a.size(); ++i) {
				diff = a[i] - b[i];
				sum += diff * diff;
			}
			return sum;
		}

		// Project to tanh space
		inline float space_proj_to_tanh(float x) const {
			return std::atanh(((x * 0.9999999f) - box_pxl_plus) * box_pxl_div);
		}

		// Project to atanh space (pixel space)
		inline float space_proj_to_pixel(float x) const {
			return std::tanh(x) * box_pxl_mul + box_pxl_plus;
		}

		inline void LoadToNN(std::vector<std::vector<float>> &modifier, std::vector<std::vector<float>> &current_attack, const std::vector<std::vector<float>> &tanh_inputs, unsigned worker_id) {
			for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
				auto &modrow = modifier[batch_idx];
				auto &tanrow = tanh_inputs[batch_idx];
				auto &catrow = current_attack[batch_idx];
				float tmp;
				for (unsigned inp_idx = 0; inp_idx != nninputs_ip.size(); ++inp_idx) {
					catrow[inp_idx] = tmp = space_proj_to_pixel(modrow[inp_idx] + tanrow[inp_idx]);
					nninputs_ip[inp_idx]->SetOwnLevel(tmp, batch_idx);
				}
			}
		}

		CarliniWagnerL2ThreadAble(const CarliniWagnerL2ThreadAble &) = delete;
		CarliniWagnerL2ThreadAble &operator=(const CarliniWagnerL2ThreadAble &) = delete;
	public:
		
		CarliniWagnerL2ThreadAble(LearnGuiderFwBPgThreadAble &model,
								  unsigned batch_size,
								  unsigned threads_count,
								  bool is_targeted = true):CarliniWagnerL2ThreadAble(model, CarliniWagnerL2Params(), batch_size, is_targeted) {}


		CarliniWagnerL2ThreadAble(LearnGuiderFwBPgThreadAble &model,
								  CarliniWagnerL2Params params,
								  unsigned batch_size,
								  unsigned threads_count,
								  bool is_targeted = true)
			: model(model), nninputs(model.GetLayers()[0]->Neurons()), nnoutputs(model.GetOutputs()),
			threads_count(threads_count),
			batch_size(batch_size),
			max_iterations(params.max_iterations),
			early_stop_hits_max(params.early_stop_chances_count),
			state_printing_period(params.state_printing_period),
			confidence(params.confidence),
			learning_rate(params.learning_rate),
			loss1_scale_init(params.loss1_scale_init),
			early_stop_severity(params.early_stop_severity),
			l2_grad_influence(params.l2_grad_influence * 2.0f),
			box_min(params.box_min),
			box_max(params.box_max),
			binary_search_steps(params.binary_search_steps),
			is_targeted(is_targeted), 
			allow_early_stop(params.allow_early_stop), 
			StatePrinter(params.StatePrinter) {
			nninputs_bp.reserve(nninputs.size());
			nninputs_ip.reserve(nninputs.size());
			for (auto elem : nninputs) {
				if (!dynamic_cast<nn::interfaces::InputNeuronI *>(elem) || !dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The input layer must consist of neurons based on interfaces::BasicBackPropogableInterface!");
				}
				nninputs_ip.push_back(dynamic_cast<interfaces::InputNeuronI *>(elem));
				nninputs_bp.push_back(dynamic_cast<interfaces::BasicBackPropogableInterface *>(elem));
			}

			nnoutputs_bp.reserve(nnoutputs.size());
			for (auto elem : nnoutputs) {
				if (!dynamic_cast<nn::interfaces::BasicBackPropogableInterface *>(elem)) {
					throw std::runtime_error("The output layer must consist of neurons based on interfaces::BasicBackPropogableInterface!");
				}
				nnoutputs_bp.push_back(dynamic_cast<interfaces::BasicBackPropogableInterface *>(elem));
			}

			// factors to map between tanh-space and original pixel bounds
			box_pxl_plus = (box_max + box_min) / 2.0f;
			box_pxl_mul = (box_max - box_min) / 2.0f;
			box_pxl_div = 1.0f / box_pxl_mul;
		}

		// Attack a single batch of inputs
		std::vector<std::vector<float>> RunAttack(const std::vector<std::vector<float>> &input_batch, const std::vector<unsigned short> &target_classes, unsigned *overall_iterations_count_return = nullptr) {
			std::vector<std::vector<float>> best_attack(batch_size); // For RVO

			if (input_batch.size() != batch_size || target_classes.size() != batch_size) {
				throw std::runtime_error("input_batch size != batch_size or target_classes size != batch_size");
			}

			model.SetThreadsCount(threads_count);

			std::vector<std::vector<float>> current_attack(batch_size, std::vector<float>(nninputs.size()));

			std::vector<std::vector<float>> tanh_inputs(batch_size, std::vector<float>(nninputs.size()));
			// Convert inputs into tanh-space so that any modifier remains in valid pixel range
			for (int i = 0; i != batch_size; ++i) {
				for (int j = 0; j != nninputs.size(); ++j) {
					tanh_inputs[i][j] = space_proj_to_tanh(input_batch[i][j]);
				}
			}

			nn::optimizers::Adam optimizer(learning_rate);
			std::vector<std::vector<nn::optimizers::Adam::State>> optimizer_states(batch_size, std::vector<nn::optimizers::Adam::State>(nninputs.size()));


			// Setup binary search bounds for the trade-off constant c
			std::vector<float> loss_scale_lower(batch_size, 0.0f),
				loss_scale_upper(batch_size, 1e10f),
				loss_scale(batch_size, loss1_scale_init);

			// To store best adversarial example per image
			std::vector<float> best_l2(batch_size, std::numeric_limits<float>::max());

			std::vector<std::vector<float>> modifier(batch_size, std::vector<float>(nninputs.size()));

			std::barrier<> threads_barrier(threads_count);

			unsigned overall_iters = 0;

			float loss_prev = std::numeric_limits<float>::max();
			unsigned loss_hits = 0;

			// Atomics for correct work
			std::atomic<float> loss;
			std::atomic_flag printer;
			std::atomic_bool do_abort = false;

			// Thread worker function
			auto AttackWorker = [&](unsigned worker_id) {
				std::vector<std::vector<float>> caches(model.GetRequiredCachesCount());

				// Outer loop: binary search over loss1_scale
				for (unsigned short bins = 0; bins != binary_search_steps; ++bins) {
					// Prepare optimization setup
					for (unsigned idx = worker_id; idx < batch_size; idx += threads_count) {
						// Initialize modifier (the variable we optimize) to zero
						std::fill(modifier[idx].begin(), modifier[idx].end(), 0.0f);

						// Reset Adam
						for (auto &stt : optimizer_states[idx]) {
							optimizer.Reset(&stt);
						}
					}

					if (worker_id == 0) {
						loss_prev = std::numeric_limits<float>::max();
						loss_hits = 0;
					}

					threads_barrier.arrive_and_wait();

					// Inner loop: minimization of loss1*c + loss2 (L2)
					for (unsigned iteration = 0;; ++iteration) {
						// Building & loading adversarial example in pixel-space
						LoadToNN(modifier, current_attack, tanh_inputs, worker_id);

						// Reset loss
						loss.store(0.0f, std::memory_order_relaxed);
						do_abort.store(false, std::memory_order_relaxed);
						printer.clear(std::memory_order_relaxed);

						threads_barrier.arrive_and_wait();

						// Forward propagation for batch
						model.WorkerDoForward(worker_id, &caches[0]);

						if (iteration == max_iterations) {
							if (worker_id == 0) {
								overall_iters += iteration;
							}
							break;
						}

						// Compute error & classification loss
						for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
							unsigned short target_class = target_classes[batch_idx];
							//    real = logit of target class
							float real = nnoutputs[target_class]->OwnLevel(batch_idx);
							//    other = max logit of non-targets
							float other = -std::numeric_limits<float>::infinity();
							unsigned short other_class = std::numeric_limits<unsigned short>::max();
							{
								float tval;
								for (unsigned short k = 0; k != nnoutputs.size(); ++k) {
									if (k != target_class) {
										tval = nnoutputs[k]->OwnLevel(batch_idx);
										if (tval > other) {
											other = tval;
											other_class = k;
										}
									}
								}
							}

							// hinge loss
							float loss1;
							if (is_targeted) {
								loss1 = std::max(0.0f, other - real + confidence);
								if (loss1 > 0.0f) {
									nnoutputs_bp[target_class]->BackPropAccumulateError(loss_scale[batch_idx], batch_idx); // +1.0f * loss_scale[batch_idx]
									nnoutputs_bp[other_class]->BackPropAccumulateError(-loss_scale[batch_idx], batch_idx); // -1.0f * loss_scale[batch_idx]
								}
							} else {
								loss1 = std::max(0.0f, real - other + confidence);
								if (loss1 > 0.0f) {
									nnoutputs_bp[other_class]->BackPropAccumulateError(loss_scale[batch_idx], batch_idx); // +1.0f * loss_scale[batch_idx]
									nnoutputs_bp[target_class]->BackPropAccumulateError(-loss_scale[batch_idx], batch_idx); // -1.0f * loss_scale[batch_idx]
								}
							}

							// total loss
							loss.fetch_add(loss1 * loss_scale[batch_idx] + l2_norm2(current_attack[batch_idx], input_batch[batch_idx]), std::memory_order_relaxed);
						}

						if (allow_early_stop) {
							threads_barrier.arrive_and_wait();

							// Dummy worker_id check due to there the all threads is "idle" because of barrier
							if (worker_id == 0) {
								// Optional early abort if loss stops improving
								float lss = loss.load(std::memory_order_relaxed);

								if (lss < loss_prev * early_stop_severity) {
									loss_prev = lss; // update baseline
									loss_hits = 0;
								} else {
									// no significant improvement
									if (++loss_hits > early_stop_hits_max) {
										do_abort.store(true, std::memory_order_relaxed);
									}
								}
							}
						}

						threads_barrier.arrive_and_wait();

						if (do_abort.load(std::memory_order_relaxed)) { // memory_order_relaxed due to guarantee from barrier
							// Reset inputs grads
							for (unsigned nrni = worker_id; nrni < nninputs_bp.size(); nrni += threads_count) {
								nninputs_bp[nrni]->BackPropResetError();
							}
							if (worker_id == 0) {
								overall_iters += iteration;
							}
							// Break the iteration loop
							break;
						}

						// Backward propagation for batch
						model.WorkerDoBackward(worker_id, &caches[0]);

						// Modifier optimization
						for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
							auto &inprow = input_batch[batch_idx];
							auto &catrow = current_attack[batch_idx];
							auto &modrow = modifier[batch_idx];
							auto &optrow = optimizer_states[batch_idx];
							float grad;
							for (unsigned inp_idx = 0; inp_idx != nninputs_bp.size(); ++inp_idx) {
								grad = nninputs_bp[inp_idx]->BackPropGetFinalError(batch_idx);
								grad += l2_grad_influence * (inprow[inp_idx] - catrow[inp_idx]); // Applying L2 norm grad: 2.0 * (inprow[inp_idx] - catrow[inp_idx])
								grad *= box_pxl_div; // Applying scale to convert form pixel to tanh space
								modrow[inp_idx] += optimizer.CalcDelta(grad, &optrow[inp_idx]);
							}
						}

						if (!printer.test_and_set(std::memory_order_acq_rel)) { // Do printing with the first idle thread
							if (StatePrinter && !(iteration % state_printing_period)) {
								StatePrinter(bins, iteration, loss.load(std::memory_order_relaxed));
							}
						}

						threads_barrier.arrive_and_wait();

						// Reset inputs grads
						for (unsigned nrni = worker_id; nrni < nninputs_bp.size(); nrni += threads_count) {
							nninputs_bp[nrni]->BackPropResetError();
						}
					}

					// Build final adversarial examples and update bests, do binary search step
					for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
						unsigned short tgt = target_classes[batch_idx];
						unsigned short pred = nn::netquality::NeuroArgmax(nnoutputs, batch_idx);
						bool success = is_targeted ? (pred == tgt) : (pred != tgt);

						// If successful
						if (success) {
							if (best_attack[batch_idx].empty()) {
								best_attack[batch_idx].resize(nninputs.size());
							}

							float l2 = l2_norm2(current_attack[batch_idx], input_batch[batch_idx]);
							if (l2 < best_l2[batch_idx]) { // If smaller L2 than previous, record as best
								best_l2[batch_idx] = l2;
								best_attack[batch_idx] = current_attack[batch_idx];
							}

							// if attack succeeded, try smaller loss_scale
							loss_scale_upper[batch_idx] = std::min(loss_scale_upper[batch_idx], loss_scale[batch_idx]);
							loss_scale[batch_idx] = (loss_scale_lower[batch_idx] + loss_scale_upper[batch_idx]) / 2.0f;
						} else {
							// if failed, increase loss_scale (or binary search if upper bound known)
							loss_scale_lower[batch_idx] = std::max(loss_scale_lower[batch_idx], loss_scale[batch_idx]);
							if (loss_scale_upper[batch_idx] < 1e9f)
								loss_scale[batch_idx] = (loss_scale_lower[batch_idx] + loss_scale_upper[batch_idx]) / 2.0f;
							else
								loss_scale[batch_idx] *= 10.0f;
						}
					}

					if (!printer.test_and_set(std::memory_order_acq_rel)) { // Do printing with the first idle thread
						if (StatePrinter) {
							StatePrinter(bins, std::numeric_limits<unsigned>::max(), loss.load(std::memory_order_relaxed));
						}
					}

					threads_barrier.arrive_and_wait();
				}
			};

			std::vector<std::thread> workers;
			for (unsigned i = 0; i != threads_count; ++i) {
				workers.emplace_back(AttackWorker, i);
			}

			for (auto &thr : workers) {
				thr.join();
			}

			if (overall_iterations_count_return) {
				*overall_iterations_count_return = overall_iters;
			}

			// Return the best adversarial examples found
			return best_attack;
		}

	};
}