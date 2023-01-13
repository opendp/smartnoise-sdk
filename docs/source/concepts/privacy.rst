#########################################
Privacy Technologies and Privacy Strategy
#########################################

Here we describe the relationship between privacy enhancing technologies and organizational privacy strategy. This is intended to be a resource for privacy professionals, to help position privacy enhancing technologies within the context of an overall privacy strategy.

First, we outline an overall framework for privacy strategy, starting with high level objectives and working down to concrete requirements.  We then discuss the role that privacy enhancing technologies play in this framework.

Organizations collect and use data about individuals in order to provide better products, services, or experiences.  However, these benefits come with the risk of harm to individuals as a result of privacy violations.  When designing a privacy strategy, your job is to prevent privacy harms while still providing the benefits of data collection and use. 

Sources of Obligations
======================

The first step in designing a privacy strategy is to identify the obligations that apply to the data collection.  Privacy obligations can be extrinsic or intrinsic, and may vary based on the type of data, location of service, and other factors.  Sources include:

Extrinsic Obligations
---------------------

Many obligations are imposed by external sources, and typically form a subset of the desired protection.

* **Statutes**.  A statue (or code) is a law coded by a legislature.  For example, the Privacy Act of 1974 governs how the US government handles data about citizens.
* **Regulations**.  Regulatory agencies are delegated broad authority to make rules to enforce statutes.  Although these rules are not coded by the legislature, they carry the force of law, and many of the concrete privacy obligations are found in regulations.
* **Contracts**.  Organizations often have contractual obligations that govern data use and protection.  For example, two organizations may agree to share data, and the agreement may include privacy obligations.
* **Policies**.  A policy is a more prescriptive document that describes how an organization will protect privacy.  These can be policies describing steps to achieve regulatory compliance, or even policies that are more specific to the organization's needs.

Intrinsic Obligations
---------------------

Most organizations will have additional privacy goals that go above and beyond those imposed by external sources. Achieving baseline regulatory compliance will provide some level of protection for users, but will not protect against all potential privacy harms.  Organizations may have additional privacy commitments that provide additional protection for all users, or for specific groups of users or types of data.

These additional commitments are typically described in internal policy documents, created by the organization consistent with its mission and values, and developed through a process of threat-modeling.  These non-statutory commitments may be communicated in whole or in part in public-facing documents such as privacy policies, but may often remain internal.

Obligation Types
================

Examples of broad types of privacy obligations include:

* **Minimize Collection**: Collect only what is necessary, and expunge when no longer needed
* **Informed Consent**: Inform users about what is collected and how it is used, get consent
* **User Control**: Allow users to control collection, request deletion, and opt-out
* **Security**: Protect data from unauthorized access, use, or disclosure
* **Anonymization**: Remove identifiers and other personally identifiable information
* **Downstream Protection**: Protect against membership inference, record linkage, and re-identification on data that has been shared or retained

Concrete Requirements
=====================

All of the above obligations are eventually mapped to concrete requirements.  For example, obligations about informed consent will translate into language published in privacy policies as well as detailed UX design to show users what is collected and how it is used.  Obligations about security will translate into technical requirements for data storage and access control, audits, and so on.  Obligations about anonymization and de-identification will translate into technical requirements for data processing.

Broadly speaking, the role for privacy technologies is greater in addressing the obligation types near the bottom of the list above (e.g. security, anonymization, and downstream protection).  The role of UX design and process is greater for the obligation types near the top of the list (e.g. informed consent, user control, and minimization).  Because this document is focused on privacy enhancing technologies, we focus here on examples of concrete requirements that are more technical.  Some examples include:

Prohibit Egress of Individual Data
----------------------------------

Concrete requirements often specify where data may or may not reside.  For example, a requirement may state that "Data may not reside on servers outside of the United States."  This is a concrete requirement that is often used to prevent data from being shared with foreign governments, or to prevent data from being shared with a foreign government without the user's consent.  

In addition to such data sovereignty rules, an organization may segregate all customer content into an "eyes-off" environment where it is not accessible by the organization's employees. There are a wide variety of architectures for such environments, but they typically prevent any human access to protected data while allowing some limited ability for the data processor to query metadata or build models. Such data products must ensure that no individual data is leaked.

More generally, this requirement arises when multiple organizations with sensitive data want to produce shared models and insights, or when silos inside a large organization want to collaborate while ensuring defense in depth.

In some cases, egress of individual-level data may be permitted so long as it is scrubbed of identifiers and other personal information.  Such relaxations are very specific to threat model and may substantially weaken privacy protections, but are common in practice.

Prevent Membership Inference
----------------------------

A company may produce a deep learning model that predicts whether or not a particular lung scan indicates a disease.  The model is trained on a dataset of patient lung scans and labels indicating whether or not the scan indicates the disease.  The model is then used to predict the disease status of a new patient.  However, the model may also be used maliciously to infer whether or not a particular patient is in the dataset.  This is a privacy harm because it allows the model to be used to infer whether or not a particular patient has the disease.  The requirement to prevent such re-identification may be implicit in regulations such as HIPAA or GDPR, or may be an explicit commitment of the organization.

The obligation to prevent membership inference exists, not only to prevent such harmful disclosure directly, but also to ensure conformance with the other obligations.  For example, if a user's data has been expunged due to retention policies or a delete request, but a downstream machine learning model can be used to infer that the user is in the dataset, then the user has not been provided with the benefit of the data minimization.  Similarly, all of the access control policies and clean room environments in the world will be for naught if a downstream model can be used to expose sensitive data.

Membership inference serves as a starting point for other privacy attacks such as database reconstruction, record linkage, and re-identification.  Therefore, concrete requirements around membership inference are often used as a proxy for other privacy harms.

Regulations are not typically prescriptive about how to prevent such inference, only specifying that organizations must provide reasonable assurances.  Because of this, the organization can make a judgment call based on a conservative interpretation of the obligation in conjunction with some threat modeling.  For example, a concrete requirement could state that "An adversary with access to the model weights cannot infer whether or not a particular patient is in the dataset with probability greater than a coin toss on average or greater than 70% in the worst case.".

Membership inference is a concern for any sort of downstream data products that are distributed more widely than the source data, or retained under different rules.  Such data products typically include aggregate reports, dashboards, data warehousing cubes, machine learning models, and so on.

This requirement is highly dependent on the threat model, and may be different for different types of data, or different types of adversaries.  For example, some models are trained on public data, or the membership of the model is publicly known, and it is the label or other sensitive data that is protected.

Privacy Enhancing Technologies
==============================

