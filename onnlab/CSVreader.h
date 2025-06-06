#pragma once
#include <fstream>
#include <string>
#include <vector>


class CSVreader {
	CSVreader(const CSVreader &) = delete;
	CSVreader &operator=(const CSVreader &) = delete;

	std::ifstream file;

	std::string header;
	std::vector<unsigned short> header_labels_locations;

	std::string row;
	std::vector<unsigned short> row_data_locations;


	inline bool FillupRow(std::string &target, std::vector<unsigned short> &locations) {
		if (!std::getline(file, target)) {
			return false;
		}
		locations.clear();
		locations.push_back(0);
		std::string::size_type pos = 0;
		while ((pos = target.find(',', pos)) != std::string::npos) {
			target[pos] = '\0';
			++pos;
			locations.push_back(static_cast<unsigned short>(pos));
		}
		return true;
	}

	inline void Init(const char *filename, bool flid) {
		file.open(filename, std::ios::in);
		if (file.fail()) {
			throw std::runtime_error("Can't open CSV file, have no access!");
		}
		if (!flid) {
			FillupRow(header, header_labels_locations); // FIrst line is header
		}
	}
public:
	CSVreader(const std::string &filename, bool first_line_is_data = false) {
		Init(filename.c_str(), first_line_is_data);
	}
	CSVreader(const char *filename, bool first_line_is_data = false) {
		Init(filename, first_line_is_data);
	}

	inline unsigned short GetHeaderItemsCount() {
		return static_cast<unsigned short>(header_labels_locations.size());
	}
	inline const char *GetHeaderItem(unsigned short index) {
		if (index < header_labels_locations.size()) {
			return &header[header_labels_locations[index]];
		}
		return nullptr;
	}

	// Fetch row & Return count of items in row
	inline unsigned short FetchNextRow() {
		if (FillupRow(row, row_data_locations)) {
			return static_cast<unsigned short>(row_data_locations.size());
		}
		row_data_locations.clear();
		return 0;
	}

	// Get cell data from fetched row
	inline const char *GetRowItem(unsigned short index) {
		if (index < row_data_locations.size()) {
			return &row[row_data_locations[index]];
		}
		return nullptr;
	}

	const char *operator[](unsigned short index) {
		return GetRowItem(index);
	}
};