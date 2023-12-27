/*!\file CArrayEnc.hpp
 * \brief
 *
 *  Created on: 13.06.2019
 *      Author: tombr
 */

#ifndef CARRAYENC_HPP_
#define CARRAYENC_HPP_

#include "CEnc.hpp"
#include "CArray.hpp"
#include "CEntry.hpp"
#include <vector>
#include <iostream>

using namespace std;

/*!
 *
 */
class CArrayEnc: public CEnc {
public:
	CArrayEnc();
	virtual ~CArrayEnc();
	int searchInTable(const string& compare); //sucht in der Tabelle nach einem passenden String, gibt -1 zurück, wenn er keinen passenden findet
	vector<unsigned int> encode(const string& a); //codiert gegebenen String nach LZW verfahren
	void printSymbolTable(unsigned int index) ;

private:
	CArray<CEntry, LZW_DICT_SIZE> m_symbolTable; //Dictonary, als Array
	int currentSize;
};

#endif /* CARRAYENC_HPP_ */
