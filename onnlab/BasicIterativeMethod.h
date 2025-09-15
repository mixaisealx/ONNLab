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
	// Settings for Basic Iterative Method (BIM) Attack / Projected Gradient Descent (PGD) Attack
	struct BasicIterativeMethodParams {
		BasicIterativeMethodParams(
			float box_min = 0.0f, // minimum pixel value
			float box_max = 1.0f, // maximum pixel value
			float confidence = 0.0f, // minimum required difference between the classes as a result
			bool allow_early_stop = true, // whether to stop early if no progress
			unsigned max_iterations = 10000, // max optimization iterations
			float early_stop_severity = 0.9999f, // severity factor of loss changes for continuation, midpoint = 1.0, lower - more severe, higher - less severe
			unsigned early_stop_chances_count = 5, // the number of "second" chances so loss can decrease sufficiently and not perform a early stop
			float learning_rate = 1e-2f // Adam learning rate
		): max_iterations(max_iterations),
			confidence(confidence),
			early_stop_chances_count(early_stop_chances_count),
			state_printing_period(100),
			learning_rate(learning_rate),
			early_stop_severity(early_stop_severity),
			box_min(box_min),
			box_max(box_max),
			allow_early_stop(allow_early_stop),
			StatePrinter(nullptr) {}
		unsigned max_iterations; // max optimization iterations
		unsigned early_stop_chances_count; // the number of "second" chances so loss can decrease sufficiently and not perform a early stop
		unsigned state_printing_period; // the iteration period of state output
		float confidence; // minimum required difference between the classes as a result
		float learning_rate; // Adam learning rate
		float early_stop_severity; // severity factor of loss changes for continuation, midpoint = 1.0, lower - more severe, higher - less severe
		float box_min; // minimum pixel value
		float box_max; // maximum pixel value
		bool allow_early_stop; // whether to stop early if no progress
		using StatePrinterT = void(*)(unsigned iteration, float loss);
		StatePrinterT StatePrinter;
	};

	// Basic Iterative Method (BIM) Attack / Projected Gradient Descent (PGD) Attack
	class BasicIterativeMethod {
		LearnGuiderFwBPg &model;
		unsigned batch_size, max_iterations, early_stop_hits_max, state_printing_period;
		float confidence, learning_rate, early_stop_severity;
		float box_min, box_max;
		bool is_targeted;
		BasicIterativeMethodParams::StatePrinterT StatePrinter;

		const std::vector<interfaces::NBI *> &nninputs, &nnoutputs;
		std::vector<interfaces::InputNeuronI *> nninputs_ip;
		std::vector<interfaces::BasicBackPropogableInterface *> nninputs_bp, nnoutputs_bp;

		BasicIterativeMethod(const BasicIterativeMethod &) = delete;
		BasicIterativeMethod &operator=(const BasicIterativeMethod &) = delete;

		inline void OptimizeInput(float &input, float delta) const {
			input -= delta;
			if (input < box_min) input = box_min;
			else if (input > box_max) input = box_max;
		}
	public:

		BasicIterativeMethod(LearnGuiderFwBPg &model,
							 unsigned batch_size,
							 bool is_targeted = true):BasicIterativeMethod(model, BasicIterativeMethodParams(), batch_size, is_targeted) {}


		BasicIterativeMethod(LearnGuiderFwBPg &model,
							 BasicIterativeMethodParams params,
							 unsigned batch_size,
							 bool is_targeted = true)
			: model(model), nninputs(model.GetLayers()[0]->Neurons()), nnoutputs(model.GetOutputs()),
			batch_size(batch_size),
			max_iterations(params.max_iterations),
			early_stop_hits_max(params.early_stop_chances_count),
			state_printing_period(params.state_printing_period),
			confidence(params.confidence),
			learning_rate(params.learning_rate),
			early_stop_severity(params.allow_early_stop ? params.early_stop_severity : std::numeric_limits<float>::infinity()),
			box_min(params.box_min),
			box_max(params.box_max),
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
		}

		// Attack a single batch of inputs
		std::vector<bool> RunAttack(std::vector<std::vector<float>> &input_batch, const std::vector<unsigned short> &target_classes, unsigned *overall_iterations_count_return = nullptr) {
			std::vector<bool> successfullness(target_classes.size(), false); // RVO

			if (input_batch.size() != batch_size || target_classes.size() != batch_size) {
				throw std::runtime_error("input_batch size != batch_size or target_classes size != batch_size");
			}

			std::vector<float> perfect_outs(nnoutputs.size());

			nn::optimizers::Adam optimizer(learning_rate);
			std::vector<std::vector<nn::optimizers::Adam::State>> optimizer_states(batch_size, std::vector<nn::optimizers::Adam::State>(nninputs.size()));

			// Reset Adam
			for (auto &bstt : optimizer_states) {
				for (auto &stt : bstt) {
					optimizer.Reset(&stt);
				}
			}

			unsigned overall_iters = 0;

			// Track previous loss for early abort
			float loss_prev = std::numeric_limits<float>::max();
			float loss = 0.0f;
			unsigned loss_hits = 0;

			for (unsigned iteration = 0;; ++iteration) {
				for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
					auto &inprow = input_batch[batch_idx];
					for (unsigned inp_idx = 0; inp_idx != nninputs_ip.size(); ++inp_idx) {
						nninputs_ip[inp_idx]->SetOwnLevel(inprow[inp_idx], batch_idx);
					}
				}

				// Forward propagation for batch
				model.DoForward();

				if (iteration == max_iterations) {
					overall_iters = iteration;
					break;
				}

				loss = 0.0f;
				// Compute error & classification loss
				for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
					std::fill(perfect_outs.begin(), perfect_outs.end(), 0.0f);

					unsigned short target_class = target_classes[batch_idx];

					if (is_targeted) { // Lets grow target class probability
						perfect_outs[target_class] = 1.0f;
					} else { // Lets find another class and grow it's probability
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
						perfect_outs[other_class] = 1.0f;
					}

					loss += model.FillupOutsError(perfect_outs, batch_idx, true);;
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
						overall_iters = iteration;
						break;
					}
				}

				// Backward propagation for batch
				model.DoBackward();

				// Input optimization
				for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
					auto &inprow = input_batch[batch_idx];
					auto &optrow = optimizer_states[batch_idx];
					for (unsigned inp_idx = 0; inp_idx != nninputs_bp.size(); ++inp_idx) {
						OptimizeInput(inprow[inp_idx], optimizer.CalcDelta(nninputs_bp[inp_idx]->BackPropGetFinalError(batch_idx), &optrow[inp_idx]));
					}
				}

				if (StatePrinter && !(iteration % state_printing_period)) {
					StatePrinter(iteration, loss);
				}

				// Reset inputs grads
				for (auto &nrn : nninputs_bp) {
					nrn->BackPropResetError();
				}
			}

			// Preparing report
			for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
				unsigned short target_class = target_classes[batch_idx];
				unsigned short real_class = nn::netquality::NeuroArgmax(nnoutputs, batch_idx);

				if (is_targeted) {
					if (target_class == real_class) {
						successfullness[batch_idx] = true;
					}
				} else {
					if (target_class != real_class) {
						successfullness[batch_idx] = true;
					}
				}
			}

			if (overall_iterations_count_return) {
				*overall_iterations_count_return = overall_iters;
			}

			return successfullness;
		}

	};

	// Basic Iterative Method (BIM) Attack / Projected Gradient Descent (PGD) Attack supporting multi-thread processing
	class BasicIterativeMethodThreadAble {
		LearnGuiderFwBPgThreadAble &model;
		unsigned threads_count, batch_size, max_iterations, early_stop_hits_max, state_printing_period;
		float confidence, learning_rate, early_stop_severity;
		float box_min, box_max;
		bool is_targeted, allow_early_stop;
		BasicIterativeMethodParams::StatePrinterT StatePrinter;

		const std::vector<interfaces::NBI *> &nninputs, &nnoutputs;
		std::vector<interfaces::InputNeuronI *> nninputs_ip;
		std::vector<interfaces::BasicBackPropogableInterface *> nninputs_bp, nnoutputs_bp;

		inline void OptimizeInput(float &input, float delta) const {
			input -= delta;
			if (input < box_min) input = box_min;
			else if (input > box_max) input = box_max;
		}

		BasicIterativeMethodThreadAble(const BasicIterativeMethodThreadAble &) = delete;
		BasicIterativeMethodThreadAble &operator=(const BasicIterativeMethodThreadAble &) = delete;
	public:

		BasicIterativeMethodThreadAble(LearnGuiderFwBPgThreadAble &model,
									   unsigned batch_size,
									   unsigned threads_count,
									   bool is_targeted = true):BasicIterativeMethodThreadAble(model, BasicIterativeMethodParams(), batch_size, is_targeted) {}


		BasicIterativeMethodThreadAble(LearnGuiderFwBPgThreadAble &model,
									   BasicIterativeMethodParams params,
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
			early_stop_severity(params.early_stop_severity),
			box_min(params.box_min),
			box_max(params.box_max),
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
		}

		// Attack a single batch of inputs
		std::vector<bool> RunAttack(std::vector<std::vector<float>> &input_batch, std::vector<unsigned short> &target_classes, unsigned *overall_iterations_count_return = nullptr) {
			std::vector<bool> successfullness(target_classes.size(), false); // RVO

			if (input_batch.size() != batch_size || target_classes.size() != batch_size) {
				throw std::runtime_error("input_batch size != batch_size or target_classes size != batch_size");
			}

			model.SetThreadsCount(threads_count);

			nn::optimizers::Adam optimizer(learning_rate);
			std::vector<std::vector<nn::optimizers::Adam::State>> optimizer_states(batch_size, std::vector<nn::optimizers::Adam::State>(nninputs.size()));

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
				
				std::vector<float> perfect_outs(nnoutputs.size());
				nn::errcalc::ErrorCalcSoftMAX softmax_calculator(nnoutputs.size());

				for (unsigned iteration = 0;; ++iteration) {
					for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
						auto &inprow = input_batch[batch_idx];
						for (unsigned inp_idx = 0; inp_idx != nninputs_ip.size(); ++inp_idx) {
							nninputs_ip[inp_idx]->SetOwnLevel(inprow[inp_idx], batch_idx);
						}
					}

					// Reset loss
					loss.store(0.0f, std::memory_order_relaxed);
					do_abort.store(false, std::memory_order_relaxed);
					printer.clear(std::memory_order_relaxed);

					threads_barrier.arrive_and_wait();

					// Forward propagation for batch
					model.WorkerDoForward(worker_id, &caches[0]);

					if (iteration == max_iterations) {
						if (worker_id == 0) {
							overall_iters = iteration;
						}
						break;
					}

					// Compute error & classification loss
					for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
						std::fill(perfect_outs.begin(), perfect_outs.end(), 0.0f);

						unsigned short target_class = target_classes[batch_idx];

						if (is_targeted) { // Lets grow target class probability
							perfect_outs[target_class] = 1.0f;
						} else { // Lets find another class and grow it's probability
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
							perfect_outs[other_class] = 1.0f;
						}

						float local_loss = model.FillupOutsError(&softmax_calculator, perfect_outs, batch_idx, true);
						// total loss
						loss.fetch_add(local_loss, std::memory_order_relaxed);
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
							overall_iters = iteration;
						}
						// Break the iteration loop
						break;
					}

					// Backward propagation for batch
					model.WorkerDoBackward(worker_id, &caches[0]);

					// Input optimization
					for (unsigned batch_idx = worker_id; batch_idx < batch_size; batch_idx += threads_count) {
						auto &inprow = input_batch[batch_idx];
						auto &optrow = optimizer_states[batch_idx];
						for (unsigned inp_idx = 0; inp_idx != nninputs_bp.size(); ++inp_idx) {
							OptimizeInput(inprow[inp_idx], optimizer.CalcDelta(nninputs_bp[inp_idx]->BackPropGetFinalError(batch_idx), &optrow[inp_idx]));
						}
					}

					if (!printer.test_and_set(std::memory_order_acq_rel)) { // Do printing with the first idle thread
						if (StatePrinter && !(iteration % state_printing_period)) {
							StatePrinter(iteration, loss.load(std::memory_order_relaxed));
						}
					}

					threads_barrier.arrive_and_wait();

					// Reset inputs grads
					for (unsigned nrni = worker_id; nrni < nninputs_bp.size(); nrni += threads_count) {
						nninputs_bp[nrni]->BackPropResetError();
					}
				}
			};

			std::vector<std::thread> workers;
			for (unsigned i = 0; i != threads_count; ++i) {
				workers.emplace_back(AttackWorker, i);
			}

			for (auto &thr : workers) {
				thr.join();
			}

			// Preparing report
			for (unsigned batch_idx = 0; batch_idx != batch_size; ++batch_idx) {
				unsigned short target_class = target_classes[batch_idx];
				unsigned short real_class = nn::netquality::NeuroArgmax(nnoutputs, batch_idx);

				if (is_targeted) {
					if (target_class == real_class) {
						successfullness[batch_idx] = true;
					}
				} else {
					if (target_class != real_class) {
						successfullness[batch_idx] = true;
					}
				}
			}

			if (overall_iterations_count_return) {
				*overall_iterations_count_return = overall_iters;
			}

			return successfullness;
		}

	};
}