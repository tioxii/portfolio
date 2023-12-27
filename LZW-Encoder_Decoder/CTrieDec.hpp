/*!\file CTrieDec.hpp
 * \brief
 *
 *  Created on: 21.06.2019
 *      Author: tombr
 */

#ifndef CTRIEDEC_HPP_
#define CTRIEDEC_HPP_

#include "CDec.hpp"
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
class CTrieDec: public CDec {
public:
	CTrieDec();
	virtual ~CTrieDec();
	string decode(const vector<unsigned int>& encoded);

private:
	CArray<CKnot, LZW_DICT_SIZE> m_symbolTable;
};

#endif /* CTRIEDEC_HPP_ */
