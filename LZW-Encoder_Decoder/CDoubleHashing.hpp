/*!\file CDoubleHashing.hpp
 * \brief
 *
 *  Created on: 18.06.2019
 *      Author: tombr
 */

#ifndef CDOUBLEHASHING_HPP_
#define CDOUBLEHASHING_HPP_

/*!
 *
 */
class CDoubleHashing {
public:
	static CDoubleHashing& getInstance();
	virtual ~CDoubleHashing();
	unsigned int hash(unsigned int I, unsigned int J, unsigned int dict_size, unsigned int attempt);

private:
	static CDoubleHashing m_instance;
	CDoubleHashing(const CDoubleHashing& other); //Kopierkonstruktor, Zuweisungsoperator und Konstruktor sind privat, damit man keine Chance hat
	CDoubleHashing operator=(const CDoubleHashing& other);
	CDoubleHashing();
};

#endif /* CDOUBLEHASHING_HPP_ */
