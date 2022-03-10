# Dynesty

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/Dynesty.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/Dynesty.jl/dev)
[![Build Status](https://github.com/ptiede/Dynesty.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/Dynesty.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/Dynesty.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/Dynesty.jl)

A Julia interface to the python nested sampling library [dynesty](https://github.com/joshspeagle/dynesty)

This is built on PyCall and imports a lot of the functionality of dynesty. There are some differences in the interface to make the code more "Julian".
