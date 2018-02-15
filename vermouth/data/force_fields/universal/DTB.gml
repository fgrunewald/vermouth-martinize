; Copyright 2018 University of Groningen
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;    http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

graph [
    node [
        id 0
        atomname "C"
    ]
    node [
        id 1
        atomname "O"
    ]
    node [
        id 2
        atomname "CA"
    ]
    node [
        id 3
        atomname "CB1"
    ]
    node [
        id 4
        atomname "CB2"
    ]
    node [
        id 5
        atomname "CG1"
    ]
    node [
        id 6
        atomname "CG2"
    ]
    node [
        id 7
        atomname "CZ"
    ]
    node [
        id 8
        atomname "SG1"
    ]
    node [
        id 9
        atomname "SG2"
    ]
    node [
        id 10
        atomname "HB1"
    ]
    node [
        id 11
        atomname "HB2"
    ]
    node [
        id 12
        atomname "HZ"
    ]
    node [
        id 13
        atomname "HS1"
    ]
    node [
        id 14
        atomname "HS2"
    ]
    edge [source 0 target 1]
    edge [source 0 target 2]
    edge [source 2 target 3]
    edge [source 2 target 4]
    edge [source 3 target 5]
    edge [source 4 target 6]
    edge [source 5 target 7]
    edge [source 6 target 7]
    edge [source 5 target 8]
    edge [source 6 target 9]
    edge [source 3 target 10]
    edge [source 4 target 11]
    edge [source 7 target 12]
    edge [source 8 target 13]
    edge [source 9 target 14]
]
