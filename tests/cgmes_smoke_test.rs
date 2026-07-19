use gridoxide::cgmes::CimDataset;

/// A minimal, self-contained CGMES fragment (not sourced from any external
/// file) exercising a scalar field, a cross-reference, and a downcast —
/// enough to prove the `cimdecoder`/`cimstructs` git dependency resolves,
/// builds, and decodes correctly. Nothing about conversion correctness yet;
/// see tests/cgmes_microgrid_be_test.rs (Phase 3) for that.
const FRAGMENT: &str = r##"<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:cim="http://iec.ch/TC57/CIM100#">
  <cim:BaseVoltage rdf:ID="BV_110">
    <cim:IdentifiedObject.name>110 kV</cim:IdentifiedObject.name>
    <cim:BaseVoltage.nominalVoltage>110</cim:BaseVoltage.nominalVoltage>
  </cim:BaseVoltage>
  <cim:TopologicalNode rdf:ID="TN_0">
    <cim:IdentifiedObject.name>Node 0</cim:IdentifiedObject.name>
    <cim:TopologicalNode.BaseVoltage rdf:resource="#BV_110" />
  </cim:TopologicalNode>
</rdf:RDF>"##;

#[test]
fn decodes_inline_fragment_and_reports_type_counts() {
    let ds = CimDataset::decode_str(FRAGMENT).expect("decode failed");

    assert_eq!(ds.by_type["BaseVoltage"].len(), 1);
    assert_eq!(ds.by_type["TopologicalNode"].len(), 1);

    let tn_mrid = &ds.by_type["TopologicalNode"][0];
    let tn = ds.entries[tn_mrid]
        .element
        .as_any()
        .downcast_ref::<cimstructs::TopologicalNode>()
        .expect("TopologicalNode downcast");
    let base_voltage_mrid = &tn.base_voltage.as_ref().expect("missing BaseVoltage ref").mrid;

    let bv = ds.entries[base_voltage_mrid]
        .element
        .as_any()
        .downcast_ref::<cimstructs::BaseVoltage>()
        .expect("BaseVoltage downcast");
    assert_eq!(bv.nominal_voltage, Some(110.0));
}
