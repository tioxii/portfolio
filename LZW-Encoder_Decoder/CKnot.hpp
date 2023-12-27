/*!\file CKnot.hpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#ifndef ENTRY_CKNOT_HPP_
#define ENTRY_CKNOT_HPP_

#include "CEntry.hpp"

/*!
 *
 */
class CKnot: public CEntry {
public:
	CKnot();
	virtual ~CKnot();
	int getParent() const;
	void setParent(int parent);
private:
	int m_parent;
};

#endif /* ENTRY_CKNOT_HPP_ */
