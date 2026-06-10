# Porject description


Hallo.

This project attempts to perform the first mean-field theoretic analysis of an irreducible topological phase. The basic setting for this project is the Watanabe bounds on the space groups allowed that allow a topological state - you an find the relevant papers in that folder. 

What we want to do is focus on a specific case - Space Group 135. This SG, via the tables in the appendix, allows, but does not require an interacting topological state (more precisely a Lieb-Schultz-Mattis-Hastings-Oshikawa theorem that does not allow long range order from short range entangled hamiltonians is shown and so if a gapped ground state with non-zero degeneracy is found here - it must be topological) in SG135 whereas this is forbidden in the non-interacting case. As a control in SG130 neither should be possible. In the non-int case, SG135 can be shown to support a model which has a Double Dirac point, and is hence ungapped. We invesrtigated this for HK models of the interaction (the paper in the theory notes) but this is HK so it is limited to that.

We want to see if a slave boson decomposition of SG135 supports this state. There is a group theoretic and a numerical question here.

1. The group theoretic question is whether there is a factorization of the electron representation into the product of a non-trivial bosonic representation with a fermionic representation (an uninteresting case would be the fermionic group = electron group, bosonic = trivial, this is what we expect). The relevant group theory is worked through in Bradely and cracknell and the Bilbao server has the references for all of this.

2. So we should first show that there are such reps for the fermion and the bosonic part (being careful about the unitary/anti-unitary nature). I think there are but you should enumerate them.

3. Then we have to propose a microscopically plausible Hamiltonian (like the one we did in the june write up ) that could support this factorization.

4. Then we need to solve the resulting self-consistency equations, 

5. And derive the phase diagram.

An analogue of this work was done for the Kane-Mele model in the theory notes, so we should start by reproducing those results. An interesting question here is whether in 2D it is possible to get the same phenomenology - i.e. are there choices of bosonic and fermionic reps which are non-trivial and the tensor product of which gives a valid electron representation (i.e fermionic) of a 2D space group.  You should be able to do this check by brute force from the BIlbao server.

Try to do the following:

1. First verify that you can solve the self-consistency equations given in the appendix of the KHM paper and check it against their phase diagram figures. This is the pre-requisite to everything, and you should not move on until you have very high confidence that you can do this,or you run out of wall clock time.

2. Then see if you can answer the representation theory in 2D. If there do exist such representations, then see if you can get the phase diagram via the analogous procedure to the KMH paper for that model. If there do not, prove it. If you prove it cannot happen, check prior literature for this result and report what you found.

3. If you finish that, or you prove it cannot happen, or you get stuck, Move onto the 3D case. I tried to do the analogue of the KMH procedure in /Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/135_MFT/notes/theory/SG135_write_up_June.pdf. I got stuck because I could not solve the self consistency equations. Solving them should be standard, just tedious via well known numerical technqiues. Try to mae all the reasonable simplifying assumptions in the SG135 case we set up (half-filling, homogenous + isotropic mean field Hubbard-Stratonovich fields etc...) and see if you can solve the SG135 case. If you can, start to relax those assumption.

4. Then enumerate the possible reps in SG135 which give rise to a non-trivial bosonic rep which factors with a fermionic rep to give the composite electron rep. Confim the proposed slave boson can host this, if it doesn't modify it so that it can. Solve the equations and derive the phase diagram.

5. Do a clear write up. Even if you fail at step 1, it is very important that someone can easily understand what you did. You should produce an executive summary of bullet points no more than 300 words, each bullet less than 30 words with an accompanying graph that demonstrates it. 

