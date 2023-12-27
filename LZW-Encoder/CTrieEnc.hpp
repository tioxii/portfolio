/*!\file CTrieEnc.hpp
 * \brief
 *
 *  Created on: 19.06.2019
 *      Author: tombr
 */

#ifndef CTRIEENC_HPP_
#define CTRIEENC_HPP_

#include "CEnc.hpp"
#include <string>
#include <vector>
#include "CDoubleHashing.hpp"
#include "CArray.hpp"
#include "CKnot.hpp"
#include "CForwardCounter.hpp"
#include <iostream>

/*!
 *
 */

class CTrieEnc: public CEnc {
public:
	CTrieEnc();
	virtual ~CTrieEnc();
	vector<unsigned int> encode(const string& a);
	string getEntry(unsigned int pos);
private:
	CArray<CKnot, LZW_DICT_SIZE> m_symbolTable;
};

#endif /* CTRIEENC_HPP_ */
