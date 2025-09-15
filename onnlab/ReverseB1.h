#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicLayerI.h"

#include <functional>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <stdexcept>

namespace nn::reverse
{
	class ReverseB1 {
	public:
		static inline bool SLE_Triangolize(std::vector<std::vector<float>> &x_matrix_augm, uint16_t rows_count, uint16_t augm_col_idx) {
			uint16_t cell_limit = std::min(rows_count, augm_col_idx);
			++augm_col_idx;
			float max_value, temp;
			uint16_t max_idx;
			for (uint16_t refcell = 0; refcell != cell_limit; ++refcell) {
				max_value = std::fabs(x_matrix_augm[refcell][refcell]);
				max_idx = refcell;
				for (uint16_t row = refcell + 1; row < rows_count; ++row) {
					temp = std::fabs(x_matrix_augm[row][refcell]);
					if (temp > max_value) {
						max_value = temp;
						max_idx = row;
					}
				}
				if (max_value < 1e-10f) {
					return false;
				}
				if (max_idx != refcell) {
					std::swap(x_matrix_augm[refcell], x_matrix_augm[max_idx]);
				}
				for (uint16_t row = refcell + 1; row < rows_count; ++row) {
					float multiplier = -x_matrix_augm[row][refcell] / x_matrix_augm[refcell][refcell];
					x_matrix_augm[row][refcell] = 0;
					for (uint16_t column = refcell + 1; column < augm_col_idx; ++column) {
						x_matrix_augm[row][column] += x_matrix_augm[refcell][column] * multiplier; //x_matrix_augm[row][augm_col_idx] += x_matrix_augm[refcell][augm_col_idx] * multiplier; - at the last stage thanks to ++augm_col_idx;
					}
				}
			}
			return true;
		}

		static inline bool SLE_TriangolizePerm(std::vector<std::vector<float>> &x_matrix_augm, const std::vector<uint16_t> &column_permutation, uint16_t augm_col_idx) {
			uint16_t rows_count = (uint16_t)x_matrix_augm.size();
			uint16_t cell_limit = std::min(rows_count, augm_col_idx);
			float max_value, temp;
			uint16_t max_idx;
			for (uint16_t refcell = 0; refcell != cell_limit; ++refcell) {
				max_value = std::fabs(x_matrix_augm[refcell][column_permutation[refcell]]);
				max_idx = refcell;
				for (uint16_t row = refcell + 1; row < rows_count; ++row) {
					temp = std::fabs(x_matrix_augm[row][column_permutation[refcell]]);
					if (temp > max_value) {
						max_value = temp;
						max_idx = row;
					}
				}
				if (max_value < 1e-10f) {
					return false;
				}
				if (max_idx != refcell) {
					std::swap(x_matrix_augm[refcell], x_matrix_augm[max_idx]);
				}
				for (uint16_t row = refcell + 1; row < rows_count; ++row) {
					float multiplier = -x_matrix_augm[row][column_permutation[refcell]] / x_matrix_augm[refcell][column_permutation[refcell]];
					x_matrix_augm[row][column_permutation[refcell]] = 0;
					x_matrix_augm[row][augm_col_idx] += x_matrix_augm[refcell][augm_col_idx] * multiplier;
					for (uint16_t column = refcell + 1; column < augm_col_idx; ++column) {
						x_matrix_augm[row][column_permutation[column]] += x_matrix_augm[refcell][column_permutation[column]] * multiplier;
					}
				}
			}
			return true;
		}

		static inline void SLE_SubstandX(std::vector<std::vector<float>> &a_matrix_augm_triangolized, uint16_t augm_col_idx, uint16_t start_column, std::vector<float> &x_vector_result) {
			float rowsumm;
			for (int32_t leftcell = start_column; leftcell >= 0; --leftcell) {
				rowsumm = 0;
				for (uint16_t column = leftcell + 1; column != augm_col_idx; ++column) {
					rowsumm += a_matrix_augm_triangolized[leftcell][column] * x_vector_result[column];
				}
				x_vector_result[leftcell] = (a_matrix_augm_triangolized[leftcell][augm_col_idx] - rowsumm) / a_matrix_augm_triangolized[leftcell][leftcell];
			}
		}

		static inline bool SLE_SolveSquare(std::vector<std::vector<float>> &a_matrix_augm, uint16_t augm_col_idx, std::vector<float> &x_vector_result) {
			if (a_matrix_augm.size() < augm_col_idx || !SLE_Triangolize(a_matrix_augm, augm_col_idx, augm_col_idx)) {
				return false;
			}
			SLE_SubstandX(a_matrix_augm, augm_col_idx, augm_col_idx - 1, x_vector_result);
			return true;
		}

		static inline void SLE_CalcEqualX(std::vector<std::vector<float>> &a_matrix_augm_triangolized, uint16_t augm_col_idx, uint16_t start_column, std::vector<float> &x_vector_result) {
			float rowsumm = 0;
			for (uint16_t column = start_column; column != augm_col_idx; ++column) {
				rowsumm += a_matrix_augm_triangolized[start_column][column];
			}
			std::fill(x_vector_result.begin() + start_column, x_vector_result.end(), a_matrix_augm_triangolized[start_column][augm_col_idx] / rowsumm);
		}

		static inline void LayerSolver_Variate(std::vector<std::vector<float>> &augmatrix, const std::vector<uint16_t> &permutation, uint16_t augmatrix_columns_count, std::function<void(const std::vector<float> &x_vector_result)> x_result_callback, uint8_t deviation_depth_exponential = 1, float initial_deviate_coefficient = 0.6f, float exponential_fade_coefficient = 0.5f) {
			if (!SLE_TriangolizePerm(augmatrix, permutation, augmatrix_columns_count)) {
				throw std::runtime_error("An unexpected singular matrix.");
			}
			uint16_t substand_cell = (uint16_t)augmatrix.size() - 1;

			std::vector<float> x_vector_result(augmatrix_columns_count);
			SLE_CalcEqualX(augmatrix, augmatrix_columns_count, substand_cell, x_vector_result);

			// An option with the equal variables can also be useful
			SLE_SubstandX(augmatrix, augmatrix_columns_count, substand_cell, x_vector_result);
			x_result_callback(x_vector_result);

			uint8_t changing_variables_count = (uint8_t)(augmatrix_columns_count - augmatrix.size());
			float variables_ref_value = x_vector_result.back(); // "mathematical expectation"

			std::vector<float> deviation_cache(deviation_depth_exponential);
			float deviator = initial_deviate_coefficient;
			for (auto &item : deviation_cache) {
				item = variables_ref_value * deviator;
				deviator *= exponential_fade_coefficient; // Exponential fade
			}

			uint16_t x_offset = (uint16_t)augmatrix.size();
			std::vector<int8_t> bitmap(changing_variables_count * deviation_depth_exponential + 1, -1);
			for (; bitmap.back() == -1; ) { // Exponent!!!
				for (uint8_t var = 0; var != changing_variables_count; ++var) {
					uint32_t bitmask_offset = var * deviation_depth_exponential;
					deviator = variables_ref_value;
					for (uint8_t devd = 0; devd != deviation_depth_exponential; ++devd) {
						deviator += deviation_cache[devd] * bitmap[bitmask_offset + devd];
					}
					x_vector_result[x_offset + var] = deviator;
				}
				SLE_SubstandX(augmatrix, augmatrix_columns_count, substand_cell, x_vector_result);
				x_result_callback(x_vector_result);

				for (auto &item : bitmap) { // Plus one
					if (item == 1) {
						item = -1;
					} else {
						item = 1;
						break;
					}
				}
			}
		}

		ReverseB1() {
		}

		void SetLayerReverseActivationFunction(nn::interfaces::BasicLayerInterface *layer, std::function<float(float)> unapply_nonlinear) {
			for (auto nrn : layer->Neurons()) {
				storage[nrn].unapply_nonlinear = unapply_nonlinear;
			}
		}

		void FillTargetExactOutputs(const std::vector<float> &outputs, nn::interfaces::BasicLayerInterface *lr_output) {
			auto oiter = outputs.begin();
			for (auto opn = lr_output->Neurons().begin(), eopn = lr_output->Neurons().end(); opn != eopn; ++opn, ++oiter) {
				storage[*opn].bp_out_value = *oiter;
			}
		}

		void ApplyLayerSolver(nn::interfaces::BasicLayerInterface *lr_input, nn::interfaces::BasicLayerInterface *lr_output) {
			uint16_t augmatrix_rows_count = (uint16_t)lr_output->Neurons().size();
			uint16_t augmatrix_columns_count = (uint16_t)lr_input->Neurons().size();

			std::vector<std::vector<float>> augmatrix(augmatrix_rows_count, std::vector<float>(augmatrix_columns_count + 1));
			{
				uint16_t row = 0, column;
				for (auto nrout : lr_output->Neurons()) {
					column = 0;
					auto &rowref = augmatrix[row];
					for (auto iconn : nrout->InputConnections()) {
						rowref[column] = iconn->Weight();
						++column;
					}
					auto &nstore = storage[nrout];
					nstore.bp_out_summ = nstore.unapply_nonlinear(nstore.bp_out_value);
					rowref[column] = nstore.bp_out_summ;
					++row;
				}
			}

			if (augmatrix_rows_count == augmatrix_columns_count) { // Can use a linear solver! There are exactly as many equations as we need.
				std::vector<float> calculated_inputs(augmatrix_columns_count);
				if (!SLE_SolveSquare(augmatrix, augmatrix_columns_count, calculated_inputs)) {
					throw std::runtime_error("An unexpected singular matrix.");
				}
				auto nriter = lr_input->Neurons().begin();
				for (auto value : calculated_inputs) {
					auto &input_neuron = storage[*nriter];
					input_neuron.bp_out_hypo.clear();
					input_neuron.bp_out_hypo.push_back(value);
					input_neuron.bp_out_hypo_type = 1; // EXACT_VALUES
					++nriter;
				}
				return; // Done!
			} else if (augmatrix_rows_count > augmatrix_columns_count) { // Redundant system of equations. We will have to go through all the combinations of equations.
				std::vector<std::vector<float>> augmatrix_window(augmatrix_columns_count);
				std::copy(augmatrix.begin(), augmatrix.begin() + augmatrix_columns_count, augmatrix_window.begin());
				
				std::vector<float> calculated_inputs(augmatrix_columns_count);
				//There we must run calc on current combo (before any permutations)
				if (!SLE_SolveSquare(augmatrix_window, augmatrix_columns_count, calculated_inputs)) {
					throw std::runtime_error("An unexpected singular matrix.");
				}
				auto input_neurons_iter = lr_input->Neurons().begin();
				for (auto value : calculated_inputs) {
					auto &input_neuron = storage[*input_neurons_iter];
					input_neuron.bp_out_hypo.clear();
					input_neuron.bp_out_hypo.push_back(value);
					input_neuron.bp_out_hypo_type = 1; // EXACT_VALUES
					++input_neurons_iter;
				}

				std::vector<uint16_t> permutation(augmatrix_rows_count);
				{
					uint16_t i = 0;
					for (auto &val : permutation) {
						val = i++;
					}
				}

				// Let's generate permutation for C^n_k set
				uint16_t subset_last_row_idx = augmatrix_columns_count - 1; // k
				uint16_t set_last_row_idx = augmatrix_rows_count - 1; // n
				uint16_t currrent_set_row = set_last_row_idx;
				uint16_t currrent_subset_row = subset_last_row_idx;
				while (subset_last_row_idx != set_last_row_idx) {
					// Do permute
					std::swap(permutation[currrent_subset_row], permutation[currrent_set_row]);
					//There we must run calc on current combo
					for (uint16_t row = 0; row != augmatrix_columns_count; ++row) {
						augmatrix_window[row] = augmatrix[permutation[row]]; // Restoring the "corrupted" rows of matrix AND filling up with new data (by permutation)
					}
					if (!SLE_SolveSquare(augmatrix_window, augmatrix_columns_count, calculated_inputs)) {
						throw std::runtime_error("An unexpected singular matrix.");
					}
					input_neurons_iter = lr_input->Neurons().begin();
					for (auto value : calculated_inputs) {
						storage[*input_neurons_iter].bp_out_hypo.push_back(value);
						++input_neurons_iter;
					}

					// Do pointers moving
					if (!currrent_subset_row) { // uint16_t subset_first_row_idx = 0;
						currrent_subset_row = subset_last_row_idx;
						--set_last_row_idx;
					}
					--currrent_set_row;
					if (currrent_set_row == subset_last_row_idx) {
						--currrent_subset_row;
						currrent_set_row = set_last_row_idx;
					}
				};
				return; // Done!
			}
			// Else... A situation where the system has infinitely many solutions. 

			std::vector<uint16_t> permutation(augmatrix_columns_count);
			{
				uint16_t i = 0;
				for (auto &val : permutation) {
					val = i++;
				}
			}
			// Let's generate permutation for C^n_k set
			uint16_t set_first_column_idx = 0;
			uint16_t subset_first_column_idx = augmatrix_rows_count - 1; // k
			uint16_t subset_last_column_idx = augmatrix_columns_count - 1; // n
			uint16_t currrent_set_column = set_first_column_idx;
			uint16_t currrent_subset_column = subset_first_column_idx;

			std::vector<std::vector<float>> augmatrix_work_copy(augmatrix);

			// Initialize storage hypo type
			for (auto &neuron : lr_input->Neurons()) {
				auto &input_neuron = storage[neuron];
				input_neuron.bp_out_hypo.clear();
				input_neuron.bp_out_hypo_type = 2; // DEVIATED_EXPECT
			}

			// Callback
			auto result_callback = [&](const std::vector<float> &x_vector_result) {
				auto input_neurons_iter = lr_input->Neurons().begin();
				for (auto value : x_vector_result) {
					storage[*input_neurons_iter].bp_out_hypo.push_back(value);
					++input_neurons_iter;
				}
			};

			//Fixate stage
			LayerSolver_Variate(augmatrix_work_copy, permutation, augmatrix_columns_count, result_callback);
			
			while (subset_first_column_idx != set_first_column_idx) {
				// Do permute
				std::swap(permutation[currrent_set_column], permutation[currrent_subset_column]);

				//Fixate stage
				augmatrix_work_copy = augmatrix;
				LayerSolver_Variate(augmatrix_work_copy, permutation, augmatrix_columns_count, result_callback);

				// Do pointers moving
				if (currrent_subset_column == subset_last_column_idx) {
					currrent_subset_column = subset_first_column_idx;
					++set_first_column_idx;
				}
				++currrent_set_column;
				if (currrent_set_column == subset_first_column_idx) {
					++currrent_subset_column;
					currrent_set_column = set_first_column_idx;
				}
			};
		}

		struct MetaData {
			float bp_out_value;
			float bp_out_summ;
			uint8_t bp_out_hypo_type; // 0 - no type, 1 - EXACT_VALUES (got from SLE solver, indexes are solution groups), 2 - DEVIATED_EXPECT (got from SLE deviator)
			std::vector<float> bp_out_hypo;
			std::function<float(float)> unapply_nonlinear;
		};

		std::unordered_map<const void *, MetaData> storage;
	};
}
