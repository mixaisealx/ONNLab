#pragma once
#include "NNBasicsInterfaces.h"
#include "AtomicSpinlock.h"

#include <limits>

namespace nn::netquality
{
	inline unsigned short NeuroArgmax(const std::vector<nn::interfaces::NBI *> &neurons, unsigned channel) {
		unsigned short maxi = 0;
		float max = -std::numeric_limits<float>::infinity();
		float curr;
		for (unsigned short idx = 0; idx != neurons.size(); ++idx) {
			curr = neurons[idx]->OwnLevel(channel);
			if (curr > max) {
				max = curr;
				maxi = idx;
			}
		}
		return maxi;
	}

	inline unsigned short VectorArgmax(const std::vector<float> &values) {
		unsigned short maxi = 0;
		float max = -std::numeric_limits<float>::infinity();
		float curr;
		for (unsigned short idx = 0; idx != values.size(); ++idx) {
			curr = values[idx];
			if (curr > max) {
				max = curr;
				maxi = idx;
			}
		}
		return maxi;
	}

	inline unsigned CalcAccurateCountOfBatch(const std::vector<nn::interfaces::NBI *> &outputs, const std::vector<std::vector<float>> &right_answers) {
		unsigned accurate = 0;
		for (unsigned batch_i = 0; batch_i != right_answers.size(); ++batch_i) {
			accurate += VectorArgmax(right_answers[batch_i]) == NeuroArgmax(outputs, batch_i);
		}
		return accurate;
	}

	inline unsigned CalcAccurateCountOfBatch(const std::vector<nn::interfaces::NBI *> &outputs, const std::vector<std::vector<float>> &right_answers, unsigned channel_init, unsigned channel_step) {
		unsigned accurate = 0;
		for (unsigned batch_i = channel_init; batch_i < right_answers.size(); batch_i += channel_step) {
			accurate += VectorArgmax(right_answers[batch_i]) == NeuroArgmax(outputs, batch_i);
		}
		return accurate;
	}

	class F1scoreMulticlassWeightsGlobal {
		std::vector<unsigned> class_numerator, class_denumerator;
		const std::vector<float> &weights;
		
		AtomicSpinlock locker;
	public:
		F1scoreMulticlassWeightsGlobal(const std::vector<float> &weights):weights(weights) {
			class_numerator.resize(weights.size());
			class_denumerator.resize(weights.size());
		}

		void AppendResult(unsigned short predicated_class, unsigned short right_class) {
			if (right_class == predicated_class) {
				class_numerator[right_class] += 2; // 2*TP (True positive)
				class_denumerator[right_class] += 2; // 2*TP (True positive)
			} else {
				class_denumerator[predicated_class] += 1; // False positive
				class_denumerator[right_class] += 1; // False negatove
			}
		}

		void AppendResults(const std::vector<nn::interfaces::NBI *> &outputs, const std::vector<std::vector<float>> &right_answers) {
			for (unsigned batch_i = 0; batch_i != right_answers.size(); ++batch_i) {
				unsigned short true_class = VectorArgmax(right_answers[batch_i]);
				unsigned short positive_class = NeuroArgmax(outputs, batch_i);
				if (true_class == positive_class) {
					class_numerator[true_class] += 2; // 2*TP (True positive)
					class_denumerator[true_class] += 2; // 2*TP (True positive)
				} else {
					class_denumerator[positive_class] += 1; // False positive
					class_denumerator[true_class] += 1; // False negatove
				}
			}
		}

		void AppendResultsThreadSafe(const std::vector<nn::interfaces::NBI *> &outputs, const std::vector<std::vector<float>> &right_answers, unsigned channel_init, unsigned channel_step) {
			for (unsigned batch_i = channel_init; batch_i < right_answers.size(); batch_i += channel_step) {
				unsigned short true_class = VectorArgmax(right_answers[batch_i]);
				unsigned short positive_class = NeuroArgmax(outputs, batch_i);
				locker.lock();
				if (true_class == positive_class) {
					class_numerator[true_class] += 2; // 2*TP (True positive)
					class_denumerator[true_class] += 2; // 2*TP (True positive)
				} else {
					class_denumerator[positive_class] += 1; // False positive
					class_denumerator[true_class] += 1; // False negatove
				}
				locker.unlock();
			}
		}

		void AppendResultsThreadSafe(const std::vector<nn::interfaces::NBI *> &outputs, const std::vector<unsigned short> &right_classes, unsigned channel_init, unsigned channel_step) {
			for (unsigned batch_i = channel_init; batch_i < right_classes.size(); batch_i += channel_step) {
				unsigned short true_class = right_classes[batch_i];
				unsigned short positive_class = NeuroArgmax(outputs, batch_i);
				locker.lock();
				if (true_class == positive_class) {
					class_numerator[true_class] += 2; // 2*TP (True positive)
					class_denumerator[true_class] += 2; // 2*TP (True positive)
				} else {
					class_denumerator[positive_class] += 1; // False positive
					class_denumerator[true_class] += 1; // False negatove
				}
				locker.unlock();
			}
		}

		float CalcF1() const {
			float f1 = 0.0f;
			for (unsigned idx = 0; idx != weights.size(); ++idx) {
				if (class_denumerator[idx]) {
					f1 += weights[idx] * class_numerator[idx] / class_denumerator[idx];
				}
			}
			return f1;
		}

		float CalcAccuracy() const {
			unsigned accurate = 0;
			unsigned overall = 0;
			for (unsigned idx = 0; idx != weights.size(); ++idx) {
				accurate += class_numerator[idx];
				overall += class_denumerator[idx];
			}
			return 100.f * accurate / overall;
		}


		float CalcF1ForClass(unsigned short target_class) const {
			return static_cast<float>(class_numerator[target_class]) / class_denumerator[target_class];
		}

		void Reset() {
			std::fill(class_numerator.begin(), class_numerator.end(), 0);
			std::fill(class_denumerator.begin(), class_denumerator.end(), 0);
		}
	};

	class ClassWeightsCalculator {
		std::vector<unsigned> samples_count;
		std::vector<float> weights;
	public:
		ClassWeightsCalculator(unsigned classes_count) {
			samples_count.resize(classes_count);
			weights.resize(classes_count);
		}

		void KeepWeightsOnly() {
			std::vector<unsigned>().swap(samples_count);
		}

		void NoteSample(unsigned short sample_class) {
			++samples_count[sample_class];
		}

		void CalcWeights() {
			unsigned count_all = 0;
			for (auto cnt : samples_count) {
				count_all += cnt;
			}
			for (unsigned idx = 0; idx != samples_count.size(); ++idx) {
				weights[idx] = samples_count[idx] / static_cast<float>(count_all);
			}
		}

		const std::vector<float> &GetWeights() const {
			return weights;
		}
	};
}