/*!\file CArrayDec.hpp
 * \brief
 *
 *  Created on: 17.06.2019
 *      Author: tombr
 */

#ifndef CARRAYDEC_HPP_
#define CARRAYDEC_HPP_

#include "CDec.hpp"
#include "CArray.hpp"
#include "CEntry.hpp"
#include <vector>
#include <iostream>

/*!
 *
 */
class CArrayDec: public CDec {
public:
	CArrayDec();
	virtual ~CArrayDec();
	int searchInTable(const string& compare);
	string decode(const vector<unsigned int>& encoded);
	void printSymbolTable(unsigned int index) ;

private:
	int currentSize;
	CArray<CEntry,LZW_DICT_SIZE> m_symbolTable;
};

#endif /* CARRAYDEC_HPP_ */
