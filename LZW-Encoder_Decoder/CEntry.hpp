/*!\file CEntry.hpp
 * \brief
 *
 *  Created on: 14.06.2019
 *      Author: tombr
 */

#ifndef CENTRY_HPP_
#define CENTRY_HPP_

#include <string>
using namespace std;

/*!
 *
 */
class CEntry {
public:
	CEntry();
	virtual ~CEntry();
	const string& getSymbol()const;
	void setSymbol(string symbol);
	static unsigned int getNumber();

private:
	string m_symbol;
	static unsigned int m_number;
};

#endif /* CENTRY_HPP_ */
